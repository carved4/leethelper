from flask import Flask, render_template, request, jsonify
import requests
import os
import re
import json
from bs4 import BeautifulSoup

app = Flask(__name__, static_folder="static")


def clean_html_content(html_content):
    """Clean HTML content and preserve formatting"""
    # Use BeautifulSoup to parse HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # Replace <pre> tags with their content plus newlines
    for pre in soup.find_all("pre"):
        pre.replace_with("\n" + pre.get_text() + "\n")

    # Replace <code> tags with their content
    for code in soup.find_all("code"):
        code.replace_with(code.get_text())

    # Replace <strong> tags with their content
    for strong in soup.find_all("strong"):
        strong.replace_with(strong.get_text())

    # Replace <em> tags with their content
    for em in soup.find_all("em"):
        em.replace_with(em.get_text())

    # Get text while preserving some formatting
    text = soup.get_text()

    # Clean up extra whitespace while preserving meaningful line breaks
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)

    return text


def get_leetcode_problem(problem_number):
    """Fetch problem details from LeetCode using GraphQL API"""
    try:
        # GraphQL endpoint
        url = "https://leetcode.com/graphql"

        # First, get the problem list
        query_for_title = """
        query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
            problemsetQuestionList: questionList(
                categorySlug: $categorySlug
                limit: $limit
                skip: $skip
                filters: $filters
            ) {
                total: totalNum
                questions: data {
                    questionId
                    title
                    titleSlug
                    difficulty
                }
            }
        }
        """

        # Variables for the first query
        variables_for_title = {
            "categorySlug": "",
            "skip": 0,
            "limit": 1,
            "filters": {"searchKeywords": str(problem_number)},
        }

        # Make the first request
        response = requests.post(
            url,
            json={"query": query_for_title, "variables": variables_for_title},
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            },
        )

        data = response.json()
        questions = (
            data.get("data", {}).get("problemsetQuestionList", {}).get("questions", [])
        )

        if not questions:
            raise Exception("Problem not found")

        # Get the first matching question's titleSlug
        title_slug = questions[0]["titleSlug"]

        # Now get the full problem details using the titleSlug
        query_for_problem = """
        query getQuestionDetail($titleSlug: String!) {
            question(titleSlug: $titleSlug) {
                questionId
                title
                difficulty
                content
                stats
                similarQuestions
                topicTags {
                    name
                    slug
                }
            }
        }
        """

        # Variables for the second query
        variables_for_problem = {"titleSlug": title_slug}

        # Make the second request
        response = requests.post(
            url,
            json={"query": query_for_problem, "variables": variables_for_problem},
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Referer": f"https://leetcode.com/problems/{title_slug}/",
            },
        )

        data = response.json()
        question = data.get("data", {}).get("question", {})

        if not question:
            raise Exception("Failed to fetch problem details")

        # Clean HTML content
        content = clean_html_content(question.get("content", ""))

        # Split content into sections
        sections = content.split("\n")
        description = []
        constraints = []
        examples = []
        current_section = description

        for line in sections:
            if "Constraints:" in line:
                current_section = constraints
            elif line.startswith("Example "):
                current_section = examples
            current_section.append(line)

        return {
            "title": question.get("title", ""),
            "difficulty": question.get("difficulty", "").capitalize(),
            "description": "\n".join(description).strip(),
            "constraints": "\n".join(constraints).strip(),
            "examples": "\n".join(examples).strip(),
            "topics": [tag.get("name") for tag in question.get("topicTags", [])],
        }

    except Exception as e:
        raise Exception(f"Failed to fetch problem: {str(e)}")


def analyze_constraints(constraints, description, title=""):
    """Analyze constraints to suggest optimal approaches"""
    analysis = {"time_complexity": "", "space_complexity": "", "approaches": []}

    # Check for specific problem patterns in the title
    title_lower = title.lower()
    if any(
        keyword in title_lower for keyword in ["trapping rain water", "trapping water"]
    ):
        # Return specific Two Pointers approach for trapping water problems
        analysis["time_complexity"] = "O(n) - single pass through the array"
        analysis["space_complexity"] = "O(1) - only using two pointers"
        analysis["approaches"] = "\n".join(
            [
                "Two Pointers Approach (Optimal for Trapping Water):",
                "1. Implementation Strategy:",
                "   • Use two pointers (left and right) starting from array ends",
                "   • Track maximum height seen from left and right",
                "   • Calculate trapped water based on the smaller of left_max and right_max",
                "",
                "2. Key Variables to Track:",
                "   • left_max: Maximum height seen from left side",
                "   • right_max: Maximum height seen from right side",
                "   • result: Total trapped water",
                "",
                "3. Algorithm Steps:",
                "   • Initialize pointers and max heights",
                "   • Move pointers based on which side has smaller height",
                "   • Update max heights and calculate trapped water",
                "   • Continue until pointers meet",
                "",
                "4. Optimization Benefits:",
                "   • Single pass through the array",
                "   • Constant extra space",
                "   • Handles all edge cases efficiently",
            ]
        )
        return analysis

    # Combine constraints and description for better pattern matching
    text_to_analyze = (constraints + " " + description).lower()

    # Define patterns dictionary with comprehensive analysis
    patterns = {
        "Two Pointers": {
            "keywords": [
                "two pointers",
                "palindrome",
                "reverse",
                "sorted array",
                "container with most water",
                "subsequence",
                "meeting points",
                "opposite ends",
                "closest",
                "three sum",
                "pair sum",
            ],
            "suggestion": "Consider using Two Pointers technique - useful for array traversal from both ends",
            "time_complexity": "O(n) - single pass through the array",
            "space_complexity": "O(1) - only using a few pointers",
            "detailed_approach": [
                "1. Identify if the problem involves:",
                "   • Finding pairs/triplets that satisfy a condition",
                "   • Processing elements from opposite ends",
                "   • Comparing elements with complementary properties",
                "2. Key implementation considerations:",
                "   • Initialize pointers at strategic positions (e.g., start/end)",
                "   • Define clear pointer movement conditions",
                "   • Handle duplicates if mentioned in constraints",
                "3. Common optimization techniques:",
                "   • Sort the array first if order doesn't matter",
                "   • Skip duplicate elements for unique results",
                "   • Use while loop for flexible pointer movement",
            ],
        },
        "Binary Search": {
            "keywords": [
                "sorted array",
                "rotated array",
                "search in",
                "find minimum",
                "find maximum",
                "binary search",
                "log(n)",
                "search range",
                "kth element",
                "median",
            ],
            "suggestion": "Apply Binary Search - reduces time complexity to O(log n), effective for sorted data",
            "time_complexity": "O(log n) - dividing search space in half each time",
            "space_complexity": "O(1) - only using a few variables for bounds",
            "detailed_approach": [
                "1. Identify binary search applicability:",
                "   • Is the data sorted or can it be sorted?",
                "   • Can we eliminate half the search space each time?",
                "   • Is there a clear target value or condition?",
                "2. Implementation strategy:",
                "   • Define clear search boundaries",
                "   • Choose appropriate mid-point calculation",
                "   • Handle edge cases (empty array, single element)",
                "3. Common variations to consider:",
                "   • Finding first/last occurrence",
                "   • Searching in rotated sorted array",
                "   • Finding peak or valley elements",
            ],
        },
        "Dynamic Programming": {
            "keywords": [
                "maximum",
                "minimum",
                "optimal",
                "longest",
                "shortest",
                "number of ways",
                "path",
                "subsequence",
                "subarray",
                "profit",
                "cost",
                "palindrome",
                "distinct",
            ],
            "suggestion": "Consider Dynamic Programming - break down into overlapping subproblems",
            "time_complexity": "Usually O(n²) or O(n*k) depending on state transitions",
            "space_complexity": "O(n) for 1D DP, O(n²) for 2D DP typically",
            "detailed_approach": [
                "1. Identify DP characteristics:",
                "   • Are there overlapping subproblems?",
                "   • Can we build solution from smaller problems?",
                "   • Is there optimal substructure?",
                "2. Design strategy:",
                "   • Define state variables clearly",
                "   • Establish base cases",
                "   • Write state transition equations",
                "3. Optimization techniques:",
                "   • Consider space optimization (rolling arrays)",
                "   • Look for redundant state variables",
                "   • Check if bottom-up is better than top-down",
            ],
        },
        "String Manipulation": {
            "keywords": [
                "string",
                "substring",
                "palindrome",
                "anagram",
                "pattern",
                "character",
                "word",
                "text",
                "concatenate",
                "reverse",
            ],
            "suggestion": "Use string manipulation techniques with careful consideration of string properties",
            "time_complexity": "O(n) for single pass, O(n*m) for pattern matching",
            "space_complexity": "Varies based on immutability requirements",
            "detailed_approach": [
                "1. String property analysis:",
                "   • Consider character set constraints",
                "   • Check for case sensitivity requirements",
                "   • Identify pattern or repetition requirements",
                "2. Implementation considerations:",
                "   • Choose between array or string methods",
                "   • Consider using string builder for concatenation",
                "   • Plan for immutability constraints",
                "3. Common techniques:",
                "   • Sliding window for substrings",
                "   • Character frequency counting",
                "   • Two pointers for palindromes",
            ],
        },
        "Math": {
            "keywords": [
                "integer",
                "number",
                "digit",
                "arithmetic",
                "calculation",
                "prime",
                "factor",
                "multiple",
                "remainder",
                "division",
            ],
            "suggestion": "Apply mathematical properties and handle edge cases carefully",
            "time_complexity": "O(log n) for digit manipulation, O(1) for fixed-size integers",
            "space_complexity": "O(1) typically for mathematical operations",
            "detailed_approach": [
                "1. Mathematical property analysis:",
                "   • Identify relevant number theory concepts",
                "   • Consider range and overflow possibilities",
                "   • Look for mathematical patterns",
                "2. Implementation strategy:",
                "   • Handle positive/negative cases separately",
                "   • Consider using modular arithmetic",
                "   • Plan for edge cases (zero, bounds)",
                "3. Common techniques:",
                "   • Digit extraction and manipulation",
                "   • Bit manipulation if applicable",
                "   • Mathematical formulas and properties",
            ],
        },
        "Sorting": {
            "keywords": [
                "sort",
                "ascending",
                "descending",
                "ordered",
                "rearrange",
                "kth largest",
                "kth smallest",
                "increasing order",
                "decreasing order",
                "rank",
            ],
            "suggestion": "Consider various sorting algorithms based on constraints",
            "time_complexity": "Ranges from O(n log n) to O(n²) depending on algorithm choice",
            "space_complexity": "O(1) for in-place sorts, O(n) for merge sort",
            "detailed_approach": [
                "1. Algorithm Selection Criteria:",
                "   • Input size and space constraints",
                "   • Stability requirements",
                "   • Presence of duplicates",
                "   • Data distribution characteristics",
                "2. Common Algorithms:",
                "   • QuickSort: O(n log n) average, in-place but unstable",
                "   • MergeSort: O(n log n) worst case, stable but O(n) space",
                "   • HeapSort: O(n log n), in-place but unstable",
                "   • Counting/Radix Sort: O(n) for specific inputs",
                "3. Implementation Considerations:",
                "   • Custom comparators",
                "   • Handling edge cases",
                "   • Memory constraints",
            ],
        },
        "Graph Traversal": {
            "keywords": [
                "graph",
                "node",
                "vertex",
                "edge",
                "path",
                "connected",
                "cycle",
                "adjacent",
                "neighbor",
                "directed",
                "undirected",
                "shortest path",
                "network",
            ],
            "suggestion": "Use graph traversal algorithms (BFS/DFS) and specialized graph algorithms",
            "time_complexity": "O(V + E) for basic traversal, varies for specialized algorithms",
            "space_complexity": "O(V) for visited set and queue/stack",
            "detailed_approach": [
                "1. Graph Representation:",
                "   • Adjacency list vs matrix",
                "   • Directed vs undirected",
                "   • Weighted vs unweighted",
                "2. Algorithm Selection:",
                "   • BFS: Shortest path in unweighted graphs",
                "   • DFS: Cycle detection, topological sort",
                "   • Dijkstra: Weighted shortest path",
                "   • Union-Find: Connected components",
                "3. Implementation Strategy:",
                "   • Track visited nodes",
                "   • Handle cycles",
                "   • Process edge cases",
            ],
        },
        "Tree Operations": {
            "keywords": [
                "tree",
                "binary tree",
                "binary search tree",
                "root",
                "leaf",
                "ancestor",
                "descendant",
                "balanced",
                "height",
                "depth",
                "traversal",
                "bst",
            ],
            "suggestion": "Apply tree traversal and manipulation techniques",
            "time_complexity": "O(n) for traversal, O(log n) for BST operations if balanced",
            "space_complexity": "O(h) for recursion stack, where h is tree height",
            "detailed_approach": [
                "1. Tree Property Analysis:",
                "   • Binary vs N-ary tree",
                "   • BST properties if applicable",
                "   • Balance requirements",
                "2. Traversal Selection:",
                "   • Inorder: Sorted order for BST",
                "   • Preorder: Copy/serialize tree",
                "   • Postorder: Delete/cleanup",
                "   • Level-order: Layer by layer",
                "3. Implementation Techniques:",
                "   • Recursive vs iterative",
                "   • Parent pointers if needed",
                "   • Stack/queue usage",
            ],
        },
        "Backtracking": {
            "keywords": [
                "combination",
                "permutation",
                "generate all",
                "possible ways",
                "valid arrangement",
                "puzzle",
                "sudoku",
                "n-queens",
                "satisfy",
                "constraint",
            ],
            "suggestion": "Use backtracking to explore all possible solutions",
            "time_complexity": "Often exponential O(b^d) where b is branching factor",
            "space_complexity": "O(d) where d is maximum recursion depth",
            "detailed_approach": [
                "1. Problem Structure:",
                "   • Define state representation",
                "   • Identify constraints",
                "   • Determine termination conditions",
                "2. Implementation Framework:",
                "   • State validation function",
                "   • Choice generation logic",
                "   • Backtracking mechanism",
                "3. Optimization Techniques:",
                "   • Prune invalid paths early",
                "   • Order choices intelligently",
                "   • Use efficient state tracking",
            ],
        },
        "Greedy": {
            "keywords": [
                "maximum",
                "minimum",
                "optimal",
                "schedule",
                "interval",
                "earliest",
                "latest",
                "profit",
                "cost",
                "local optimal",
            ],
            "suggestion": "Apply greedy strategy making locally optimal choices",
            "time_complexity": "Usually O(n log n) due to initial sorting",
            "space_complexity": "O(1) or O(n) depending on implementation",
            "detailed_approach": [
                "1. Greedy Choice Property:",
                "   • Verify local optimal leads to global",
                "   • Identify safe moves",
                "   • Consider sorting benefit",
                "2. Implementation Strategy:",
                "   • Sort if helpful",
                "   • Process in optimal order",
                "   • Track running results",
                "3. Validation:",
                "   • Prove correctness",
                "   • Consider counter-examples",
                "   • Handle edge cases",
            ],
        },
        "Sliding Window": {
            "keywords": [
                "subarray",
                "substring",
                "consecutive",
                "window",
                "contiguous",
                "fixed size",
                "at most k",
                "at least k",
                "longest",
                "shortest",
            ],
            "suggestion": "Use sliding window technique for array/string subsequence problems",
            "time_complexity": "O(n) - single pass through the array/string",
            "space_complexity": "O(1) for fixed window, O(k) for variable window",
            "detailed_approach": [
                "1. Window Type Selection:",
                "   • Fixed size window",
                "   • Variable size window",
                "   • Multiple windows",
                "2. Implementation Strategy:",
                "   • Initialize window bounds",
                "   • Define window condition",
                "   • Handle window updates",
                "3. Optimization Techniques:",
                "   • Early termination",
                "   • Efficient condition checking",
                "   • Space optimization",
            ],
        },
        "Divide and Conquer": {
            "keywords": [
                "divide",
                "merge",
                "partition",
                "half",
                "middle",
                "recursive",
                "subproblem",
                "binary",
                "segment",
                "range",
            ],
            "suggestion": "Break problem into smaller subproblems, solve and combine",
            "time_complexity": "Often O(n log n) or O(log n) depending on division strategy",
            "space_complexity": "O(log n) to O(n) depending on recursion depth",
            "detailed_approach": [
                "1. Problem Division:",
                "   • Identify division point",
                "   • Ensure subproblems are similar",
                "   • Handle base cases",
                "2. Solution Strategy:",
                "   • Recursive vs iterative approach",
                "   • Combining subproblem solutions",
                "   • Handling edge cases",
                "3. Optimization Considerations:",
                "   • Memoization if applicable",
                "   • Tail recursion",
                "   • Space-time tradeoffs",
            ],
        },
        "Bit Manipulation": {
            "keywords": [
                "bit",
                "binary",
                "AND",
                "OR",
                "XOR",
                "shift",
                "mask",
                "power of two",
                "set bit",
                "binary representation",
            ],
            "suggestion": "Use bitwise operations for optimization",
            "time_complexity": "O(1) to O(log n) depending on number of bits",
            "space_complexity": "O(1) - typically constant space",
            "detailed_approach": [
                "1. Bit Operations:",
                "   • Basic operations (AND, OR, XOR)",
                "   • Bit shifting",
                "   • Bit counting",
                "2. Common Techniques:",
                "   • Power of 2 checking",
                "   • Setting/clearing bits",
                "   • Bit masks",
                "3. Implementation Tips:",
                "   • Handle negative numbers",
                "   • Consider overflow",
                "   • Use built-in functions",
            ],
        },
        "Union Find": {
            "keywords": [
                "disjoint set",
                "connected components",
                "union",
                "find",
                "group",
                "connection",
                "merge sets",
                "equivalence",
                "partition",
            ],
            "suggestion": "Use Union-Find data structure for set operations",
            "time_complexity": "Nearly O(1) for operations with path compression",
            "space_complexity": "O(n) for storing the sets",
            "detailed_approach": [
                "1. Data Structure Setup:",
                "   • Initialize parent array",
                "   • Implement rank/size tracking",
                "   • Path compression",
                "2. Core Operations:",
                "   • Find operation with path compression",
                "   • Union operation with rank",
                "   • Connected component tracking",
                "3. Optimization Techniques:",
                "   • Path compression",
                "   • Union by rank/size",
                "   • Iterative find",
            ],
        },
        "Trie": {
            "keywords": [
                "prefix",
                "dictionary",
                "word search",
                "autocomplete",
                "string search",
                "word",
                "vocabulary",
                "dictionary",
                "prefix tree",
            ],
            "suggestion": "Use Trie data structure for prefix-based string operations",
            "time_complexity": "O(m) for operations, where m is key length",
            "space_complexity": "O(ALPHABET_SIZE * m * n) for n keys",
            "detailed_approach": [
                "1. Structure Design:",
                "   • Node representation",
                "   • Character mapping",
                "   • End of word marking",
                "2. Core Operations:",
                "   • Insert words",
                "   • Search words/prefixes",
                "   • Delete words",
                "3. Optimization Options:",
                "   • Compressed tries",
                "   • Memory-efficient nodes",
                "   • Character set optimization",
            ],
        },
        "Heap": {
            "keywords": [
                "k largest",
                "k smallest",
                "k closest",
                "top k",
                "priority",
                "running median",
                "stream",
                "minimum element",
                "maximum element",
                "sort k",
            ],
            "suggestion": "Use Heap/Priority Queue for k-element problems or streaming data",
            "time_complexity": "O(log k) per operation, O(n log k) for processing n elements",
            "space_complexity": "O(k) for storing k elements in heap",
            "detailed_approach": [
                "1. Heap Selection:",
                "   • Min-heap vs Max-heap",
                "   • K-size maintenance",
                "   • Custom comparators",
                "2. Implementation Strategy:",
                "   • Heap initialization",
                "   • Element processing order",
                "   • Size maintenance",
                "3. Optimization Techniques:",
                "   • Early termination",
                "   • Lazy deletion",
                "   • Batch processing",
                "4. Common Applications:",
                "   • K-th largest/smallest",
                "   • Median finding",
                "   • Merge k sorted lists",
            ],
        },
        "Hash Table": {
            "keywords": [
                "frequency",
                "count",
                "unique",
                "duplicate",
                "pair",
                "map",
                "dictionary",
                "set",
                "hash",
                "lookup",
            ],
            "suggestion": "Use hash table for O(1) lookups and frequency counting",
            "time_complexity": "O(n) for building, O(1) for lookups",
            "space_complexity": "O(n) for storing n elements",
            "detailed_approach": [
                "1. Data Structure Selection:",
                "   • HashMap vs HashSet",
                "   • Counter for frequencies",
                "   • Multi-map if needed",
                "2. Implementation Strategy:",
                "   • Choose appropriate key-value pairs",
                "   • Handle collisions",
                "   • Consider space-time tradeoffs",
                "3. Common Applications:",
                "   • Two-sum type problems",
                "   • Frequency counting",
                "   • Caching results",
            ],
        },
        "Stack and Queue": {
            "keywords": [
                "stack",
                "queue",
                "last element",
                "first element",
                "push",
                "pop",
                "peek",
                "parentheses",
                "brackets",
                "monotonic",
                "deque",
                "LIFO",
                "FIFO",
            ],
            "suggestion": "Use stack/queue for processing elements in specific order",
            "time_complexity": "O(1) for push/pop operations",
            "space_complexity": "O(n) for storing n elements",
            "detailed_approach": [
                "1. Structure Selection:",
                "   • Stack: LIFO operations",
                "   • Queue: FIFO operations",
                "   • Deque: Both ends access",
                "2. Common Applications:",
                "   • Parentheses matching",
                "   • Monotonic stack problems",
                "   • BFS/DFS implementations",
                "3. Implementation Tips:",
                "   • Consider edge cases",
                "   • Handle empty structure",
                "   • Track additional state if needed",
            ],
        },
    }

    # Check for specific number/integer manipulation patterns
    if any(word in text_to_analyze for word in ["integer", "number", "digit"]):
        range_match = re.search(r"-?(\d+)\s*<=\s*\w+\s*<=\s*(\d+)", constraints)
        if range_match:
            max_abs = max(
                abs(int(range_match.group(1))), abs(int(range_match.group(2)))
            )
            analysis["time_complexity"] = (
                f"O(log n) - where n is the input number (approximately {len(str(max_abs))} digits)"
            )
            analysis["space_complexity"] = (
                "O(1) - only using a few variables for calculation"
            )
            analysis["approaches"] = "\n".join(
                [
                    "1. Problem Decomposition:",
                    "   • Break down the number into individual digits",
                    "   • Consider the significance of each digit's position",
                    "   • Plan for sign handling (positive/negative)",
                    "",
                    "2. Implementation Strategy:",
                    "   • Choose between mathematical or string-based approach",
                    "   • Consider trade-offs between readability and performance",
                    "   • Plan digit processing order (left-to-right vs right-to-left)",
                    "",
                    "3. Edge Cases to Consider:",
                    "   • Integer overflow/underflow scenarios",
                    "   • Zero and negative number handling",
                    "   • Maximum/minimum value boundaries",
                    "",
                    "4. Optimization Opportunities:",
                    "   • Early termination conditions",
                    "   • Efficient digit extraction methods",
                    "   • Memory-efficient calculations",
                ]
            )
            return analysis

    # Check each pattern against the combined text
    matched_approaches = []
    for approach, pattern_info in patterns.items():
        for keyword in pattern_info["keywords"]:
            if re.search(r"\b" + re.escape(keyword) + r"\b", text_to_analyze):
                matched_approaches.append(
                    {
                        "name": approach,
                        "details": pattern_info["detailed_approach"],
                        "time": pattern_info["time_complexity"],
                        "space": pattern_info["space_complexity"],
                    }
                )
                break

    if matched_approaches:
        # Combine detailed approaches
        analysis["approaches"] = "\n\n".join(
            f"{approach['name']} Approach:\n" + "\n".join(approach["details"])
            for approach in matched_approaches
        )

        # Combine complexity analysis
        analysis["time_complexity"] = "Complexity analysis by approach:\n" + "\n".join(
            f"• {approach['name']}:\n  Time: {approach['time']}"
            for approach in matched_approaches
        )

        analysis["space_complexity"] = "Space complexity considerations:\n" + "\n".join(
            f"• {approach['name']}:\n  Space: {approach['space']}"
            for approach in matched_approaches
        )
    else:
        # Generic problem-solving framework
        analysis["approaches"] = "\n".join(
            [
                "1. Problem Understanding:",
                "   • Break down the problem requirements",
                "   • Identify input/output patterns",
                "   • List edge cases and constraints",
                "",
                "2. Solution Strategy:",
                "   • Consider brute force approach first",
                "   • Look for patterns in examples",
                "   • Identify optimization opportunities",
                "",
                "3. Implementation Planning:",
                "   • Choose appropriate data structures",
                "   • Plan error handling and validation",
                "   • Consider code organization",
                "",
                "4. Optimization Considerations:",
                "   • Time-space complexity trade-offs",
                "   • Early termination conditions",
                "   • Code readability vs performance",
            ]
        )

    return analysis


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_problem():
    try:
        problem_number = request.form.get("problem_number")
        if not problem_number:
            return jsonify({"error": "Problem number is required"}), 400

        # Fetch problem details
        problem = get_leetcode_problem(problem_number)

        if not problem:
            return jsonify({"error": "Problem not found"}), 404

        # Analyze problem and generate optimization suggestions
        optimization = analyze_constraints(
            problem["constraints"],
            problem["description"],
            problem["title"],  # Pass the title to analyze_constraints
        )

        return jsonify(
            {
                "title": problem["title"],
                "difficulty": problem["difficulty"],
                "description": problem["description"],
                "constraints": problem["constraints"],
                "examples": problem["examples"],
                "optimization": optimization,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
