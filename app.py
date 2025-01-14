from flask import Flask, render_template, request, jsonify
import requests
import os
import re
import json
from bs4 import BeautifulSoup

app = Flask(__name__, static_folder="static")

patterns_dict = {
    "two_pointers": {
        "suggestion": "Use two pointers technique for array/string traversal",
        "time_complexity": "O(n) for single pass",
        "space_complexity": "O(1) - only using pointers",
        "detailed_approach": [
            "1. Initialize two pointers (typically at start/end)",
            "2. Move pointers based on conditions",
            "3. Process elements at pointer positions",
            "4. Update pointers accordingly",
        ],
    },
    "sliding_window": {
        "suggestion": "Use sliding window for subarray/substring problems",
        "time_complexity": "O(n) for single pass",
        "space_complexity": "O(1) for fixed window, O(k) for variable",
        "detailed_approach": [
            "1. Initialize window bounds",
            "2. Expand window by moving right pointer",
            "3. Contract window from left if needed",
            "4. Update result during window modifications",
        ],
    },
    "binary_search": {
        "suggestion": "Use binary search for sorted data or search space reduction",
        "time_complexity": "O(log n)",
        "space_complexity": "O(1)",
        "detailed_approach": [
            "1. Define search space with bounds",
            "2. Calculate mid point",
            "3. Compare and adjust bounds",
            "4. Repeat until convergence",
        ],
    },
    "dynamic_programming": {
        "suggestion": "Use dynamic programming for optimal substructure problems",
        "time_complexity": "Usually O(n²) or O(n*k)",
        "space_complexity": "O(n) for 1D DP, O(n²) for 2D DP",
        "detailed_approach": [
            "1. Define state and transitions",
            "2. Create DP table/memoization",
            "3. Initialize base cases",
            "4. Fill DP table using recurrence",
            "5. Return final state or reconstruct",
        ],
    },
    "graph": {
        "suggestion": "Use graph algorithms for connected component problems",
        "time_complexity": "O(V + E) for traversal",
        "space_complexity": "O(V) for visited set",
        "detailed_approach": [
            "1. Create adjacency representation",
            "2. Choose traversal method (BFS/DFS)",
            "3. Track visited nodes",
            "4. Process nodes as needed",
        ],
    },
    "heap": {
        "suggestion": "Use heap for k-element or priority problems",
        "time_complexity": "O(n log k) for k-element processing",
        "space_complexity": "O(k) for heap storage",
        "detailed_approach": [
            "1. Initialize heap with comparator",
            "2. Process elements maintaining heap",
            "3. Extract top elements as needed",
            "4. Update heap efficiently",
        ],
    },
    "k_sum": {
        "suggestion": "Use sorting and two pointers for k-sum problems",
        "time_complexity": "O(n^(k-1)) for k-sum",
        "space_complexity": "O(1) excluding output",
        "detailed_approach": [
            "1. Sort array if needed",
            "2. Fix k-2 elements with loops",
            "3. Use two pointers for remaining sum",
            "4. Handle duplicates carefully",
        ],
    },
    "string_manipulation": {
        "suggestion": "Use string operations and character processing",
        "time_complexity": "O(n) for single pass",
        "space_complexity": "O(n) for new string creation",
        "detailed_approach": [
            "1. Process characters systematically",
            "2. Use appropriate string methods",
            "3. Handle edge cases and boundaries",
            "4. Consider string immutability",
        ],
    },
    "tree_traversal": {
        "suggestion": "Use tree traversal techniques (DFS/BFS)",
        "time_complexity": "O(n) for full traversal",
        "space_complexity": "O(h) for recursion stack",
        "detailed_approach": [
            "1. Choose traversal type",
            "2. Implement recursive/iterative",
            "3. Process nodes during traversal",
            "4. Handle null nodes properly",
        ],
    },
    "hash_table": {
        "suggestion": "Use hash table for O(1) lookups",
        "time_complexity": "O(n) for building, O(1) lookups",
        "space_complexity": "O(n) for storage",
        "detailed_approach": [
            "1. Choose key-value structure",
            "2. Handle collisions if needed",
            "3. Process elements efficiently",
            "4. Consider load factor",
        ],
    },
    "stack": {
        "suggestion": "Use stack for LIFO processing",
        "time_complexity": "O(n) for processing",
        "space_complexity": "O(n) for stack storage",
        "detailed_approach": [
            "1. Initialize stack structure",
            "2. Push/pop elements as needed",
            "3. Track stack state",
            "4. Handle empty stack cases",
        ],
    },
    "greedy": {
        "suggestion": "Use greedy approach for local optimal choices",
        "time_complexity": "Usually O(n log n) with sorting",
        "space_complexity": "O(1) or O(n)",
        "detailed_approach": [
            "1. Sort if beneficial",
            "2. Make locally optimal choices",
            "3. Prove global optimality",
            "4. Handle edge cases",
        ],
    },
    "union_find": {
        "suggestion": "Use union-find for disjoint sets",
        "time_complexity": "O(α(n)) amortized",
        "space_complexity": "O(n) for parent array",
        "detailed_approach": [
            "1. Initialize parent array",
            "2. Implement find with compression",
            "3. Union by rank/size",
            "4. Process connections",
        ],
    },
    "bit_manipulation": {
        "suggestion": "Use bitwise operations",
        "time_complexity": "O(1) to O(log n)",
        "space_complexity": "O(1)",
        "detailed_approach": [
            "1. Identify bit operations needed",
            "2. Use masks and shifts",
            "3. Handle edge cases",
            "4. Consider bit properties",
        ],
    },
    "prefix_sum": {
        "suggestion": "Use prefix sums for range queries",
        "time_complexity": "O(n) build, O(1) query",
        "space_complexity": "O(n) for prefix array",
        "detailed_approach": [
            "1. Build prefix sum array",
            "2. Handle range queries",
            "3. Consider cumulative properties",
            "4. Watch for overflow",
        ],
    },
    "intervals": {
        "suggestion": "Sort and process intervals sequentially",
        "time_complexity": "O(n log n) for sorting",
        "space_complexity": "O(n) for results",
        "detailed_approach": [
            "1. Sort intervals appropriately",
            "2. Process in order",
            "3. Handle overlaps",
            "4. Track boundaries",
        ],
    },
    "matrix": {
        "suggestion": "Use matrix traversal techniques",
        "time_complexity": "O(m*n) for full traversal",
        "space_complexity": "O(1) in-place or O(m*n)",
        "detailed_approach": [
            "1. Choose traversal pattern",
            "2. Handle boundaries",
            "3. Process elements systematically",
            "4. Consider direction arrays",
        ],
    },
    "monotonic_stack": {
        "suggestion": "Use monotonic stack for next/prev problems",
        "time_complexity": "O(n) amortized",
        "space_complexity": "O(n) for stack",
        "detailed_approach": [
            "1. Maintain monotonic property",
            "2. Process elements in order",
            "3. Handle stack operations",
            "4. Build result array",
        ],
    },
    "trie": {
        "suggestion": "Use trie for prefix-based operations",
        "time_complexity": "O(L) for word length",
        "space_complexity": "O(N*L) for N words",
        "detailed_approach": [
            "1. Design trie structure",
            "2. Implement insert/search",
            "3. Track word endings",
            "4. Optimize space usage",
        ],
    },
    "divide_and_conquer": {
        "suggestion": "Break problem into smaller subproblems",
        "time_complexity": "Often O(n log n)",
        "space_complexity": "O(log n) to O(n)",
        "detailed_approach": [
            "1. Divide problem logically",
            "2. Solve subproblems",
            "3. Combine results",
            "4. Handle base cases",
        ],
    },
    "recursion": {
        "suggestion": "Use recursive approach with base cases",
        "time_complexity": "Varies with branching",
        "space_complexity": "O(h) for recursion depth",
        "detailed_approach": [
            "1. Define base cases",
            "2. Implement recursive step",
            "3. Ensure progress",
            "4. Consider stack depth",
        ],
    },
    "bitmask": {
        "suggestion": "Use bitmask for state compression",
        "time_complexity": "O(2^n) for all states",
        "space_complexity": "O(1) per state",
        "detailed_approach": [
            "1. Design state representation",
            "2. Implement bit operations",
            "3. Track state changes",
            "4. Handle transitions",
        ],
    },
    "binary_tree": {
        "suggestion": "Use binary tree properties",
        "time_complexity": "O(n) traversal, O(log n) BST",
        "space_complexity": "O(h) for height",
        "detailed_approach": [
            "1. Choose traversal method",
            "2. Handle tree properties",
            "3. Process nodes properly",
            "4. Consider balance",
        ],
    },
    "topological_sort": {
        "suggestion": "Use topological sort for DAG ordering",
        "time_complexity": "O(V + E)",
        "space_complexity": "O(V) for visited and result",
        "detailed_approach": [
            "1. Build adjacency structure",
            "2. Track in-degrees",
            "3. Process nodes in order",
            "4. Detect cycles",
        ],
    },
}


def clean_html_content(html_content):
    """Clean HTML content and preserve formatting"""
    if not html_content:
        return ""

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
    lines = text.splitlines()  # Use splitlines() instead of split('\n')
    text = "\n".join(line.strip() for line in lines if line.strip())

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


def analyze_constraints(constraints, description, title="", problem_number=""):
    """Analyze constraints to suggest optimal approaches"""
    analysis = {"time_complexity": "", "space_complexity": "", "approaches": []}

    # First try to get pre-analyzed patterns from our database
    try:
        with open("training/pattern_labels.json", "r") as f:
            pattern_labels = json.load(f)

        # Find the problem in our database
        for problem in pattern_labels:
            if problem.get("questionId") == str(
                problem_number
            ):  # Convert to string for comparison
                # Get patterns and topic tags
                patterns = problem.get("patterns", [])
                topic_tags = [
                    tag if isinstance(tag, str) else tag.get("name")
                    for tag in problem.get("topicTags", [])
                ]

                # Get approaches for each pattern
                approaches = []
                for pattern in patterns:
                    if (
                        pattern in patterns_dict
                    ):  # We need to define patterns_dict at the top
                        pattern_info = patterns_dict[pattern]
                        approaches.append(
                            {
                                "name": pattern,
                                "suggestion": pattern_info["suggestion"],
                                "time_complexity": pattern_info["time_complexity"],
                                "space_complexity": pattern_info["space_complexity"],
                                "detailed_approach": pattern_info["detailed_approach"],
                            }
                        )

                if approaches:
                    # Combine all approaches
                    analysis["approaches"] = "\n\n".join(
                        f"{approach['name']} Approach:\n"
                        + f"Suggestion: {approach['suggestion']}\n"
                        + "\n".join(approach["detailed_approach"])
                        for approach in approaches
                    )

                    # Combine complexity analysis
                    analysis["time_complexity"] = (
                        "Time Complexity by Pattern:\n"
                        + "\n".join(
                            f"• {approach['name']}: {approach['time_complexity']}"
                            for approach in approaches
                        )
                    )

                    analysis["space_complexity"] = (
                        "Space Complexity by Pattern:\n"
                        + "\n".join(
                            f"• {approach['name']}: {approach['space_complexity']}"
                            for approach in approaches
                        )
                    )

                    return analysis

    except Exception as e:
        print(f"Error reading from pattern database: {e}")

    # If we didn't find the problem or couldn't get approaches, fall back to rule-based analysis
    return analyze_patterns_rule_based(constraints, description, title)


def analyze_patterns_rule_based(constraints, description, title):
    """Original rule-based analysis function"""
    analysis = {"time_complexity": "", "space_complexity": "", "approaches": []}

    # Move all the original pattern matching logic here
    # ... (copy all the original pattern matching code here)

    return analysis


def get_leetcode_profile(username):
    """Fetch LeetCode profile data using GraphQL API"""
    try:
        url = "https://leetcode.com/graphql"

        # GraphQL query for user profile
        query = """
        query getUserProfile($username: String!) {
            matchedUser(username: $username) {
                username
                submitStats: submitStatsGlobal {
                    acSubmissionNum {
                        difficulty
                        count
                        submissions
                    }
                }
                problemsSolvedBeatsStats {
                    difficulty
                    percentage
                }
                profile {
                    ranking
                    reputation
                    starRating
                }
            }
        }
        """

        # Variables for the query
        variables = {"username": username}

        # Make the request
        response = requests.post(
            url,
            json={"query": query, "variables": variables},
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            },
        )

        data = response.json()
        user_data = data.get("data", {}).get("matchedUser")

        if not user_data:
            raise Exception("User not found")

        # Get nested data with defaults
        submit_stats = user_data.get("submitStats", {})
        submission_numbers = submit_stats.get("acSubmissionNum", [])
        beats_stats = user_data.get("problemsSolvedBeatsStats", [])
        profile = user_data.get("profile", {})

        # Create default submission stats if empty
        if not submission_numbers:
            submission_numbers = [
                {"difficulty": "Easy", "count": 0, "submissions": 0},
                {"difficulty": "Medium", "count": 0, "submissions": 0},
                {"difficulty": "Hard", "count": 0, "submissions": 0},
            ]

        # Create default beats stats if empty
        if not beats_stats:
            beats_stats = [
                {"difficulty": "Easy", "percentage": 0},
                {"difficulty": "Medium", "percentage": 0},
                {"difficulty": "Hard", "percentage": 0},
            ]

        # Ensure all required fields exist with default values
        profile_data = {
            "username": user_data.get("username", username),
            "submitStats": {
                "acSubmissionNum": [
                    {
                        "difficulty": item.get("difficulty", "Unknown"),
                        "count": item.get("count", 0),
                        "submissions": item.get("submissions", 0),
                    }
                    for item in submission_numbers
                ]
            },
            "problemsSolvedBeatsStats": [
                {
                    "difficulty": item.get("difficulty", "Unknown"),
                    "percentage": item.get("percentage", 0),
                }
                for item in beats_stats
            ],
            "profile": {
                "ranking": profile.get("ranking", 0),
                "reputation": profile.get("reputation", 0),
                "starRating": profile.get("starRating", 0),
            },
        }

        return profile_data

    except Exception as e:
        raise Exception(f"Failed to fetch profile: {str(e)}")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_problem():
    try:
        problem_number = request.form.get("problem_number")
        if not problem_number:
            return jsonify({"error": "Problem number is required"}), 400

        # First fetch problem details from LeetCode
        problem = get_leetcode_problem(problem_number)
        if not problem:
            return jsonify({"error": "Problem not found"}), 404

        # Then analyze with our enhanced system
        optimization = analyze_constraints(
            problem["constraints"],
            problem["description"],
            problem["title"],
            problem_number,
        )

        # Return both LeetCode details and our analysis
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


@app.route("/profile", methods=["POST"])
def get_profile():
    try:
        username = request.form.get("username")
        if not username:
            return jsonify({"error": "Username is required"}), 400

        # Fetch profile data
        profile_data = get_leetcode_profile(username)

        return jsonify(profile_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
