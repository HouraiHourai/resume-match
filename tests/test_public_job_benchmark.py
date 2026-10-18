import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location('public_benchmark',
    Path(__file__).resolve().parents[1]/'scripts/public_job_benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_chinese_tokenization_preserves_technical_terms():
    tokens = benchmark.tokens('C++ Java开发工程师')
    assert 'c++' in tokens and 'java' in tokens
    assert '开发' in tokens and '工程' in tokens and '程师' in tokens
    assert 'java开' not in tokens


def test_rrf_weight_endpoints_and_missing_lexical_evidence():
    vector = np.array([.9, .8, .7])
    lexical = np.array([0., 5., 2.])
    assert benchmark.fused_order(vector, lexical, 1) == [0, 1, 2]
    assert benchmark.fused_order(vector, lexical, 0) == [1, 2, 0]


def test_paired_bootstrap_constant_difference():
    result = benchmark.bootstrap_delta([.1, .2, .3], [.2, .3, .4])
    assert np.allclose(result['paired_bootstrap_95ci'], [.1, .1])
