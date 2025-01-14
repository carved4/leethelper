import json
from collections import Counter


def analyze_patterns():
    # Load pattern mapping for comparison
    with open("pattern_mapping.json", "r") as f:
        pattern_mapping = json.load(f)

    # Load pattern labels
    with open("pattern_labels.json", "r") as f:
        pattern_labels = json.load(f)

    # Initialize sets for unique values
    all_patterns = set()
    all_topic_tags = set()

    # Counters for frequency analysis
    pattern_counter = Counter()
    tag_counter = Counter()

    # Analyze each problem
    for problem in pattern_labels:
        patterns = problem.get("patterns", [])
        topic_tags = problem.get("topicTags", [])

        # Handle both string and dict formats for topic tags
        processed_tags = []
        for tag in topic_tags:
            if isinstance(tag, dict):
                processed_tags.append(tag["name"])
            else:
                processed_tags.append(tag)

        # Add to sets
        all_patterns.update(patterns)
        all_topic_tags.update(processed_tags)

        # Add to counters
        pattern_counter.update(patterns)
        tag_counter.update(processed_tags)

    # Compare with pattern_mapping
    mapped_patterns = set(pattern_mapping.keys())
    missing_patterns = mapped_patterns - all_patterns
    extra_patterns = all_patterns - mapped_patterns

    # Print analysis
    print("=== Pattern Analysis ===")
    print(f"\nTotal unique patterns found: {len(all_patterns)}")
    print(f"Total unique topic tags found: {len(all_topic_tags)}")

    print("\nMost common patterns:")
    for pattern, count in pattern_counter.most_common():
        print(f"- {pattern}: {count} problems")

    print("\nMost common topic tags:")
    for tag, count in tag_counter.most_common():
        print(f"- {tag}: {count} problems")

    print("\n=== Pattern Mapping Comparison ===")
    if missing_patterns:
        print("\nPatterns in mapping but not in problems:")
        for pattern in missing_patterns:
            print(f"- {pattern}")

    if extra_patterns:
        print("\nPatterns in problems but not in mapping:")
        for pattern in extra_patterns:
            print(f"- {pattern}")

    print("\nAll unique topic tags:")
    for tag in sorted(all_topic_tags):
        print(f"- {tag}")


if __name__ == "__main__":
    analyze_patterns()
