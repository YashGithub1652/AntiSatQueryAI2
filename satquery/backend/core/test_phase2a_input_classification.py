from satquery.backend.core.agent import _classify_input_structure, node_classify_task


def make_image(modality, sensor="unknown", date=None):
    return {
        "modality": modality,
        "sensor": sensor,
        "metadata": {
            "acquisition_date": date,
        },
        "array": 1,
    }


def classify_task(query, images):
    state = {
        "query": query,
        "images": images,
        "query_parse": {"intent": "UNKNOWN"},
        "trace_log": [],
    }
    return node_classify_task(state)["task_type"]


def main():
    tests = [
        (
            "single optical",
            [make_image("optical", "sentinel_2")],
            "SINGLE_VQA",
            None,
        ),
        (
            "single sar",
            [make_image("sar", "sentinel_1")],
            "SINGLE_VQA",
            None,
        ),
        (
            "optical temporal dated",
            [
                make_image("optical", "sentinel_2", "2024-01-01"),
                make_image("optical", "sentinel_2", "2024-06-01"),
            ],
            "BI_TEMPORAL_CHANGE",
            None,
        ),
        (
            "optical pair no dates with change query",
            [
                make_image("optical", "sentinel_2"),
                make_image("optical", "sentinel_2"),
            ],
            "BI_TEMPORAL_CHANGE",
            None,
        ),
        (
            "optical pair no dates ambiguous",
            [
                make_image("optical", "sentinel_2"),
                make_image("optical", "sentinel_2"),
            ],
            "CLARIFICATION_NEEDED",
            None,
        ),
        (
            "optical sar",
            [
                make_image("optical", "sentinel_2"),
                make_image("sar", "sentinel_1"),
            ],
            "CROSS_MODAL_SAR_OPTICAL",
            None,
        ),
        (
            "sar sar no dates",
            [
                make_image("sar", "sentinel_1"),
                make_image("sar", "sentinel_1"),
            ],
            "CLARIFICATION_NEEDED",
            None,
        ),
        (
            "temporal reversed dates",
            [
                make_image("optical", "sentinel_2", "2024-06-01"),
                make_image("optical", "sentinel_2", "2024-01-01"),
            ],
            "BI_TEMPORAL_CHANGE",
            None,
        ),
        (
            "three images",
            [
                make_image("optical", "sentinel_2"),
                make_image("optical", "sentinel_2"),
                make_image("optical", "sentinel_2"),
            ],
            "CLARIFICATION_NEEDED",
            None,
        ),
    ]

    passed = 0

    for name, images, expected_task, _ in tests:
        structure = _classify_input_structure({"images": images})
        actual_task = classify_task(
            "What changed between these images?"
            if "change" in name or "temporal" in name
            else "Analyze these images",
            images,
        )

        task_ok = actual_task == expected_task

        print(
            f"[{'PASS' if task_ok else 'FAIL'}] "
            f"{name}: structure={structure['structure']} "
            f"task={actual_task} expected={expected_task}"
        )

        if task_ok:
            passed += 1

    print()
    print(f"RESULT: {passed}/{len(tests)} tests passed")

    if passed != len(tests):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

