import json
from typing import Dict, List, Tuple


class ApproachGenerator:
    def __init__(self):
        self.pattern_labels = self.load_json("pattern_labels.json")
        self.pattern_mapping = self.load_json("pattern_mapping.json")
        self.approaches_db = {}

    def load_json(self, filename: str) -> Dict:
        with open(filename, "r") as f:
            return json.load(f)

    def get_complexity(self, patterns: List[str], tags: List[str]) -> Tuple[str, str]:
        """Enhanced complexity analysis based on patterns and tags."""
        time_complexity = "O(n)"  # default
        space_complexity = "O(1)"  # default

        complexity_rules = {
            "sorting": ("O(n log n)", "O(1)"),
            "hash_table": ("O(n)", "O(n)"),
            "dynamic_programming": ("O(n²)", "O(n)"),
            "backtracking": ("O(2^n)", "O(n)"),
            "heap": ("O(n log n)", "O(n)"),
            "binary_search": ("O(log n)", "O(1)"),
            "two_pointers": ("O(n)", "O(1)"),
            "sliding_window": ("O(n)", "O(1)"),
            "graph": ("O(V + E)", "O(V)"),
            "tree_traversal": ("O(n)", "O(h)"),  # h is height of tree
            "bit_manipulation": ("O(1)", "O(1)"),
            "union_find": ("O(α(n))", "O(n)"),  # α is inverse Ackermann function
            "prefix_sum": ("O(n)", "O(n)"),
            "intervals": ("O(n log n)", "O(n)"),
            "greedy": ("O(n log n)", "O(1)"),  # often involves sorting
            "string_manipulation": ("O(n)", "O(n)"),
            "k_sum": ("O(n^(k-1))", "O(n)"),  # k is the sum count (e.g., 3sum, 4sum)
            "stack": ("O(n)", "O(n)"),
        }

        # Get the worst-case complexity
        for pattern in patterns:
            if pattern in complexity_rules:
                t, s = complexity_rules[pattern]
                # Compare complexities and take the worse one
                time_complexity = self.compare_complexity(time_complexity, t)
                space_complexity = self.compare_complexity(space_complexity, s)

        return time_complexity, space_complexity

    def compare_complexity(self, c1: str, c2: str) -> str:
        """Compare two complexities and return the worse one."""
        order = {
            "O(1)": 1,
            "O(log n)": 2,
            "O(n)": 3,
            "O(n log n)": 4,
            "O(n²)": 5,
            "O(2^n)": 6,
            "O(n!)": 7,
        }

        # Handle special cases
        if "V + E" in c1 or "V + E" in c2:
            return "O(V + E)"  # Graph complexity
        if "α(n)" in c1 or "α(n)" in c2:
            return "O(α(n))"  # Union-find complexity

        # Extract base complexity for comparison
        c1_base = c1.split()[0] if c1.split() else c1
        c2_base = c2.split()[0] if c2.split() else c2

        return c1 if order.get(c1_base, 0) > order.get(c2_base, 0) else c2

    def generate_approach_steps(
        self, patterns: List[str], tags: List[str]
    ) -> List[str]:
        """Generate detailed approach steps based on patterns and tags."""
        steps = []

        pattern_approaches = {
            "two_pointers": [
                "1. Initialize two pointers (typically at start/end of array)",
                "2. Move pointers based on conditions",
                "3. Process elements at pointer positions",
                "4. Update pointers accordingly",
            ],
            "sliding_window": [
                "1. Initialize window bounds (left and right pointers)",
                "2. Expand window by moving right pointer",
                "3. Contract window from left if condition is violated",
                "4. Update result during window modifications",
            ],
            "binary_search": [
                "1. Define search space with left and right bounds",
                "2. Calculate mid point and evaluate condition",
                "3. Adjust search space based on condition",
                "4. Repeat until convergence or solution found",
            ],
            "dynamic_programming": [
                "1. Define the state and state transitions",
                "2. Create DP table or memoization structure",
                "3. Initialize base cases",
                "4. Fill DP table using optimal substructure",
                "5. Return final state or reconstruct solution",
            ],
            "backtracking": [
                "1. Define base cases for recursion",
                "2. Identify choice points and constraints",
                "3. Make choices and backtrack when needed",
                "4. Track valid solutions during recursion",
            ],
            "graph": [
                "1. Create adjacency list/matrix representation",
                "2. Choose traversal method (BFS/DFS)",
                "3. Track visited nodes to avoid cycles",
                "4. Process nodes according to problem requirements",
            ],
            "tree_traversal": [
                "1. Choose traversal type (inorder/preorder/postorder)",
                "2. Implement recursive or iterative approach",
                "3. Process nodes during traversal",
                "4. Handle edge cases (null nodes)",
            ],
            "heap": [
                "1. Initialize heap with appropriate comparison function",
                "2. Add/remove elements maintaining heap property",
                "3. Access top element for current minimum/maximum",
                "4. Update heap as needed during processing",
            ],
            "hash_table": [
                "1. Define appropriate hash function and structure",
                "2. Handle collisions if necessary",
                "3. Store and retrieve key-value pairs",
                "4. Update table based on problem requirements",
            ],
            "bit_manipulation": [
                "1. Identify bit operations needed (AND, OR, XOR)",
                "2. Use bit masks for specific operations",
                "3. Handle edge cases and sign bits",
                "4. Optimize using bit manipulation properties",
            ],
            "k_sum": [
                "1. Sort array if not sorted (for k > 2)",
                "2. Use (k-2) nested loops for k > 2",
                "3. Use two pointers for the innermost loop",
                "4. Handle duplicates carefully",
                "5. Optimize by skipping duplicate values",
            ],
            "prefix_sum": [
                "1. Initialize prefix sum array/variable",
                "2. Calculate cumulative sum at each index",
                "3. Use formula: rangeSum(i,j) = prefixSum[j] - prefixSum[i-1]",
                "4. Handle edge cases (empty array, single element)",
            ],
            "intervals": [
                "1. Sort intervals by start/end time",
                "2. Process intervals in order",
                "3. Track overlaps/merges using previous interval",
                "4. Handle edge cases (nested intervals, complete overlaps)",
            ],
            "sorting": [
                "1. Choose appropriate sorting algorithm",
                "2. Consider stability requirements",
                "3. Handle duplicate elements",
                "4. Optimize for specific input characteristics",
            ],
            "greedy": [
                "1. Sort or organize data if needed",
                "2. Make locally optimal choice at each step",
                "3. Prove that local optimum leads to global optimum",
                "4. Handle edge cases that might break greedy choice",
            ],
            "stack": [
                "1. Initialize stack data structure",
                "2. Define push/pop conditions",
                "3. Process elements maintaining stack property",
                "4. Handle stack empty/full conditions",
            ],
            "union_find": [
                "1. Initialize parent array/map",
                "2. Implement find operation with path compression",
                "3. Implement union operation with rank/size",
                "4. Process connections/relationships",
                "5. Optimize using union by rank",
            ],
        }

        for pattern in patterns:
            if pattern in pattern_approaches:
                steps.extend(pattern_approaches[pattern])
                steps.append("")  # Add spacing between approaches

        return (
            steps
            if steps
            else ["No specific approach steps available for given patterns"]
        )

    def generate_approaches(self):
        """Generate approaches for all problems."""
        approaches_db = {}

        for problem_id, data in self.pattern_labels.items():
            patterns = data.get("patterns", [])
            tags = data.get("topicTags", [])

            time_complexity, space_complexity = self.get_complexity(patterns, tags)
            approach_steps = self.generate_approach_steps(patterns, tags)

            approaches_db[problem_id] = {
                "title": data["title"],
                "patterns": patterns,
                "tags": tags,
                "time_complexity": time_complexity,
                "space_complexity": space_complexity,
                "approach_steps": approach_steps,
            }

        # Save the generated approaches
        with open("generated_approaches.json", "w") as f:
            json.dump(approaches_db, f, indent=2)


if __name__ == "__main__":
    generator = ApproachGenerator()
    generator.generate_approaches()
