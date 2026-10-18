"""Low-token real structured-output smoke; injected faults reported separately."""
import argparse
import asyncio
import json
from pathlib import Path
import time
from pydantic import BaseModel
from resume_match.config import get_llm
from resume_match.services import structured_llm as _structured


class SkillResult(BaseModel):
    language: str


CALL_LIMIT = 8
calls = []


class BudgetStop(BaseException):
    pass


class Client:
    def __init__(self, name):
        self.name = name

    def with_structured_output(self, schema, method):
        # Explicitly small output and no hidden SDK retry. Production outer
        # retry/fallback logic remains unchanged. Not full production prompts.
        model = get_llm(self.name, temperature=0).model_copy(
            update={'max_tokens':128, 'max_retries':0, 'request_timeout':20})
        chain = model.with_structured_output(schema, method=method, include_raw=True)
        name = self.name

        class Invocation:
            async def ainvoke(self, prompt):
                if len(calls) >= CALL_LIMIT:
                    raise BudgetStop()
                row = {'model':name, 'status':'started', 'usage':None}
                calls.append(row)
                start = time.perf_counter()
                try:
                    response = await chain.ainvoke(prompt)
                    raw = response.get('raw')
                    row['usage'] = getattr(raw, 'usage_metadata', None)
                    parsed = response.get('parsed')
                    row['status'] = 'structured_success' if parsed is not None else 'invalid_structure'
                    return parsed
                except Exception as exc:
                    row['status'] = type(exc).__name__
                    row['http_status'] = getattr(exc, 'status_code', None)
                    # Do not expose credentials or provider error bodies.
                    raise RuntimeError(row['status']) from None
                finally:
                    row['elapsed_ms'] = (time.perf_counter()-start)*1000
        return Invocation()


class Fault:
    def __init__(self):
        self.count = 0

    async def ainvoke(self, prompt):
        self.count += 1
        raise TimeoutError('deliberately injected primary failure')


async def main(out: Path):
    if out.exists():
        raise RuntimeError('Choose a new output; preserve old evidence')
    rows=[]
    factory = _structured.get_llm
    _structured.get_llm = lambda name:Client(name)
    try:
        for language in ['Python', 'Java', 'Go']:
            start_index=len(calls)
            prompt=f'从岗位要求提取编程语言，只返回一个语言名称。岗位要求：熟悉{language}。'
            try:
                result=await _structured.ainvoke_structured_with_retry(
                    Client('deepseek').with_structured_output(SkillResult,'function_calling'),
                    prompt, schema_class=SkillResult)
                valid=result.language.casefold()==language.casefold()
            except Exception:
                valid=False
            task_calls=calls[start_index:]
            primary_success=any(c['model']=='deepseek' and c['status']=='structured_success' for c in task_calls)
            rows.append({'type':'natural', 'expected':language, 'primary_structured_success':primary_success,
                         'final_content_correct':valid, 'calls':task_calls})
            print(f'Natural {language}: primary={primary_success}, final_correct={valid}',flush=True)
        fault = Fault()
        start_index = len(calls)
        try:
            result=await _structured.ainvoke_structured_with_retry(fault,
                '从岗位要求提取编程语言，只返回一个语言名称。岗位要求：熟悉Python。',schema_class=SkillResult)
            success=result.language.casefold()=='python'
        except Exception:
            success=False
        rows.append({'type':'injected_primary_timeout', 'simulated_primary_attempts':fault.count,
                     'real_fallback_content_correct':success,'calls':calls[start_index:]})
        print(f'Injected failure: attempts={fault.count}, real_fallback_correct={success}',flush=True)
    except BudgetStop:
        rows.append({'type':'budget_stop','note':'Stopped at 8 HTTP logical invocations; no further API calls.'})
    finally:
        _structured.get_llm=factory
    known=[c['usage'] for c in calls if c['usage']]
    summary={'scope':'Tiny real structured-output smoke, not production JD/Agent generation success-rate benchmark',
             'max_output_tokens_per_call':128,'call_limit':CALL_LIMIT,'calls':calls,'rows':rows,
             'known_total_tokens':sum(u.get('total_tokens',0) for u in known),
             'calls_without_usage':sum(c['usage'] is None for c in calls)}
    out.mkdir(parents=True)
    (out/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'api_calls':len(calls),'known_total_tokens':summary['known_total_tokens'],
                      'calls_without_usage':summary['calls_without_usage']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory for the API test results')
    args = parser.parse_args()
    asyncio.run(main(args.output))

