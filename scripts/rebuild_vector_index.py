"""重建向量索引，添加精排所需字段

作用:
- 从 SQLite 读取所有历史记录
- 重新生成向量并写入 Chroma（包含新增的 hard_skills, domain, seniority 字段）
- 保证粗排精排能正常工作
"""
from resume_match.history.storage import HistoryStorage
from resume_match.rag.vector_store import get_vector_store


def rebuild_index():
    """重建向量索引"""
    storage = HistoryStorage()
    store = get_vector_store()

    # 先获取所有 ID
    entries = list(storage.iter_entries())

    if not entries:
        print("Warning: 没有历史记录")
        return

    print(f"Data: 找到 {len(entries)} 条历史记录")
    print("Processing: 开始重建向量索引...\n")

    success_count = 0
    skip_count = 0

    for i, entry in enumerate(entries, 1):
        # 获取完整记录（包含 full_result）
        full_entry = entry
        if not full_entry:
            print(f"  [{i}/{len(entries)}] Skip ID={entry.id} (无法获取)")
            skip_count += 1
            continue

        # 检查是否有完整的 JD 数据
        jd = (full_entry.full_result or {}).get("jd")
        if not jd:
            print(f"  [{i}/{len(entries)}] Skip ID={entry.id} (无 JD 数据)")
            skip_count += 1
            continue

        # 重新索引
        store.index_entry(full_entry)
        success_count += 1

        # 显示进度
        role_title = jd.get("role_title", "未知岗位")
        print(f"  [{i}/{len(entries)}] OK {role_title}")

    print("\nDone: 重建完成:")
    print(f"   - 成功: {success_count} 条")
    print(f"   - 跳过: {skip_count} 条")
    print(f"   - 总数: {store.count()} 条")


if __name__ == "__main__":
    rebuild_index()
