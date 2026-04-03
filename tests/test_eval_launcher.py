"""Tests for the parallel evaluation launcher utilities."""

import pytest
from scripts.launch_eval import resolve_tasks, partition_tasks


class TestResolveTasks:
    def test_resolve_tasks_single_set(self):
        tasks = resolve_tasks(["atomic_seen"])
        assert len(tasks) == 18
        assert "CloseBlenderLid" in tasks
        assert "TurnOnSinkFaucet" in tasks

    def test_resolve_tasks_all_target(self):
        tasks = resolve_tasks(["atomic_seen", "composite_seen", "composite_unseen"])
        assert len(tasks) == 50

    def test_resolve_tasks_dedup(self):
        tasks_with_dup = resolve_tasks(["target50", "atomic_seen"])
        tasks_no_dup = resolve_tasks(["target50"])
        assert tasks_with_dup == tasks_no_dup
        assert len(tasks_with_dup) == 50

    def test_resolve_tasks_invalid(self):
        with pytest.raises(KeyError):
            resolve_tasks(["nonexistent_task_set"])


class TestPartitionTasks:
    @pytest.fixture
    def all_tasks(self):
        return resolve_tasks(["atomic_seen", "composite_seen", "composite_unseen"])

    def test_partition_no_overlap(self, all_tasks):
        partitions = partition_tasks(all_tasks, 8)
        for i, p1 in enumerate(partitions):
            for j, p2 in enumerate(partitions):
                if i != j:
                    assert set(p1).isdisjoint(set(p2)), f"Partitions {i} and {j} overlap: {set(p1) & set(p2)}"

    def test_partition_full_coverage(self, all_tasks):
        partitions = partition_tasks(all_tasks, 8)
        union = []
        for p in partitions:
            union.extend(p)
        assert sorted(union) == sorted(all_tasks)

    def test_partition_balanced_by_horizon(self, all_tasks):
        from robocasa.utils.dataset_registry_utils import get_task_horizon

        partitions = partition_tasks(all_tasks, 8)
        loads = [sum(get_task_horizon(t) for t in p) for p in partitions]
        ratio = max(loads) / min(loads)
        assert ratio <= 2.0, f"Imbalanced partitions: max/min ratio = {ratio:.2f}"

    def test_partition_deterministic(self, all_tasks):
        p1 = partition_tasks(all_tasks, 8)
        p2 = partition_tasks(all_tasks, 8)
        assert p1 == p2

    def test_partition_more_workers_than_tasks(self):
        tasks = resolve_tasks(["atomic_seen"])
        with pytest.raises(ValueError, match="n_workers.*>.*n_tasks"):
            partition_tasks(tasks, 100)

    def test_partition_single_worker(self, all_tasks):
        partitions = partition_tasks(all_tasks, 1)
        assert len(partitions) == 1
        assert sorted(partitions[0]) == sorted(all_tasks)
