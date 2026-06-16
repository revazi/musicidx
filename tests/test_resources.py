from musicidx.resources import (
    SystemResources,
    recommend_indexing_plan,
    resolve_embedding_batch_size,
    resolve_tag_batch_size,
    resolve_worker_count,
)


def gb(value: int) -> int:
    return value * 1024**3


def test_auto_profile_uses_low_defaults_under_16gb():
    plan = recommend_indexing_plan(
        resources=SystemResources(cpu_count=8, total_memory_bytes=gb(8))
    )

    assert plan.effective_profile == "low"
    assert plan.basic_workers == 1
    assert plan.tag_workers == 1
    assert plan.embedding_batch_size == 8
    assert plan.tag_batch_size == 3


def test_auto_profile_uses_balanced_defaults_on_midrange_machine():
    plan = recommend_indexing_plan(
        resources=SystemResources(cpu_count=8, total_memory_bytes=gb(32))
    )

    assert plan.effective_profile == "balanced"
    assert plan.basic_workers == 2
    assert plan.tag_workers == 1
    assert plan.embedding_batch_size == 16
    assert plan.tag_batch_size == 5


def test_auto_profile_uses_full_defaults_only_on_large_machine():
    plan = recommend_indexing_plan(
        resources=SystemResources(cpu_count=12, total_memory_bytes=gb(64))
    )

    assert plan.effective_profile == "full"
    assert plan.basic_workers == 4
    assert plan.tag_workers == 1
    assert plan.embedding_batch_size == 32
    assert plan.tag_batch_size == 10


def test_unknown_memory_falls_back_to_low_defaults():
    plan = recommend_indexing_plan(
        resources=SystemResources(cpu_count=16, total_memory_bytes=None)
    )

    assert plan.effective_profile == "low"
    assert plan.basic_workers == 1
    assert plan.warning is not None


def test_resolve_auto_workers_by_kind():
    resources = SystemResources(cpu_count=8, total_memory_bytes=gb(32))

    assert resolve_worker_count("auto", kind="basic", resources=resources) == 2
    assert resolve_worker_count("auto", kind="tags", resources=resources) == 1
    assert resolve_worker_count("3", kind="basic", resources=resources) == 3


def test_resolve_auto_embedding_batch_size():
    resources = SystemResources(cpu_count=8, total_memory_bytes=gb(32))

    assert resolve_embedding_batch_size("auto", resources=resources) == 16
    assert resolve_embedding_batch_size("4", resources=resources) == 4


def test_resolve_auto_tag_batch_size():
    resources = SystemResources(cpu_count=8, total_memory_bytes=gb(32))

    assert resolve_tag_batch_size("auto", resources=resources) == 5
    assert resolve_tag_batch_size("2", resources=resources) == 2
