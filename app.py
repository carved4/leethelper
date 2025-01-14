from flask import Flask, render_template, request, jsonify
import requests
import os
import re
import json
from bs4 import BeautifulSoup

app = Flask(__name__, static_folder="static")

patterns_dict = {
    "two_pointers": {
        "suggestion": "Employ the two-pointer technique for efficient array/string traversal, especially useful in sorted arrays or linked lists.",
        "time_complexity": "O(n) for a single pass, where n is the length of the input.",
        "space_complexity": "O(1) - constant space is used, as only two pointers are maintained.",
        "detailed_approach": [
            "1. Initialization: Start with two pointers, often one at the beginning (left) and one at the end (right) of the array or list.",
            "2. Movement Strategy: Define conditions for moving the pointers. In a sorted array, you might move the left pointer rightward to increase the sum of the elements pointed to, and the right pointer leftward to decrease it.",
            "3. Element Processing: At each step, process the elements at the pointer positions. This could involve comparing them, adding them to a result set, or checking for a specific condition.",
            "4. Pointer Updates: Based on the processing outcome, update the pointers. For instance, if the current sum is less than a target, move the left pointer; if it's greater, move the right pointer.",
            "5. Termination: Continue until the pointers meet or cross, or until a specific condition is met, ensuring all relevant pairs or combinations are considered.",
            "Example Use Case: Finding a pair of numbers in a sorted array that adds up to a specific target value.",
        ],
    },
    "sliding_window": {
        "suggestion": "Utilize the sliding window technique for problems involving subarrays or substrings, especially when looking for a range that satisfies certain conditions.",
        "time_complexity": "O(n) for a single pass, where n is the length of the input.",
        "space_complexity": "O(1) for a fixed-size window, O(k) for a variable-size window, where k is the maximum size of the window.",
        "detailed_approach": [
            "1. Initialization: Define a window with two pointers, typically both starting at the beginning of the array or string.",
            "2. Expansion: Expand the window by moving the right pointer to include more elements. This is usually done to find a valid window that satisfies the problem's conditions.",
            "3. Contraction: If the window becomes invalid or a better solution can be found by reducing its size, contract the window by moving the left pointer.",
            "4. Result Update: During the expansion and contraction, keep track of the best or required result based on the problem's criteria.",
            "5. Iteration: Continue expanding and contracting the window until the right pointer has traversed the entire input.",
            "Example Use Case: Finding the longest substring without repeating characters or the smallest subarray with a sum greater than a given value.",
        ],
    },
    "binary_search": {
        "suggestion": "Apply binary search for problems involving sorted data or when the search space can be systematically reduced by half.",
        "time_complexity": "O(log n), where n is the size of the search space.",
        "space_complexity": "O(1) - constant space is used.",
        "detailed_approach": [
            "1. Define Search Space: Establish the initial bounds of the search space, typically with 'start' and 'end' indices in a sorted array.",
            "2. Calculate Mid Point: Compute the middle index as `mid = (start + end) // 2`. Ensure to handle potential overflow if indices are very large.",
            "3. Comparison: Compare the element at the mid index with the target value. This comparison determines how the search space will be adjusted.",
            "4. Adjust Bounds: If the target is less than the mid element, narrow the search to the left half by setting `end = mid - 1`. If the target is greater, narrow to the right half by setting `start = mid + 1`.",
            "5. Iteration: Repeat steps 2-4 until the target is found or the bounds converge (start > end), indicating the target is not present.",
            "Example Use Case: Searching for a specific value in a sorted array, finding the square root of a number, or determining the insertion point for a new element in a sorted sequence.",
        ],
    },
    "dynamic_programming": {
        "suggestion": "Employ dynamic programming for problems that exhibit optimal substructure and overlapping subproblems, allowing for efficient solutions by storing and reusing subproblem results.",
        "time_complexity": "Typically O(n²) or O(nk), depending on the number of states and transitions, where n is the problem size and k is a constraint.",
        "space_complexity": "O(n) for 1D DP arrays, O(n²) for 2D DP tables, depending on the problem's state space.",
        "detailed_approach": [
            "1. Identify Optimal Substructure: Determine if the problem can be broken down into smaller subproblems whose solutions can be combined to solve the larger problem.",
            "2. Define State and Transitions: Define the state of the DP (e.g., `dp[i]` represents the solution up to index i) and the recurrence relation that describes how states transition from one to another.",
            "3. Create DP Table/Memoization: Set up a data structure (array, table, or memoization cache) to store the results of subproblems to avoid redundant calculations.",
            "4. Initialize Base Cases: Fill in the initial values of the DP table that correspond to the simplest subproblems (e.g., `dp[0]` or `dp[1]` often have straightforward solutions).",
            "5. Fill DP Table Using Recurrence: Iteratively or recursively fill in the DP table by applying the recurrence relation, building up solutions from the base cases.",
            "6. Return Final State or Reconstruct Solution: Once the DP table is filled, the solution to the original problem is typically found in the last entry of the table or can be reconstructed by tracing back through the table.",
            "Example Use Case: Calculating Fibonacci numbers, finding the shortest path in a weighted graph, optimizing resource allocation, and solving sequence alignment problems.",
        ],
    },
    "graph": {
        "suggestion": "Apply graph algorithms for problems that can be modeled as graphs, especially those involving connected components, paths, or network structures.",
        "time_complexity": "O(V + E) for traversal algorithms like BFS and DFS, where V is the number of vertices and E is the number of edges.",
        "space_complexity": "O(V) for storing visited nodes in traversal, plus space for the graph representation (adjacency list or matrix).",
        "detailed_approach": [
            "1. Create Adjacency Representation: Convert the problem into a graph by representing it as an adjacency list or matrix. Nodes (vertices) represent entities, and edges represent relationships or connections between them.",
            "2. Choose Traversal Method: Select an appropriate traversal algorithm based on the problem's requirements. Breadth-First Search (BFS) is suitable for finding shortest paths in unweighted graphs, while Depth-First Search (DFS) is useful for exploring all paths or detecting cycles.",
            "3. Track Visited Nodes: Use a data structure (e.g., a set or array) to keep track of visited nodes during traversal to avoid revisiting them and prevent infinite loops.",
            "4. Process Nodes as Needed: During traversal, perform the necessary operations on each node. This could involve checking properties, updating values, or adding nodes to a result set.",
            "5. Handle Graph Properties: Consider specific graph properties like directed/undirected edges, weighted edges, cycles, and connectivity when implementing the algorithm.",
            "Example Use Case: Finding connected components in a social network, determining the shortest path between two locations, detecting cycles in a dependency graph, and solving network flow problems.",
        ],
    },
    "heap": {
        "suggestion": "Utilize heaps (priority queues) for problems that involve finding the k smallest or largest elements, managing priorities, or performing efficient element insertion and extraction.",
        "time_complexity": "O(n log k) for processing n elements and maintaining a heap of size k; O(log n) for individual heap operations (insertion, deletion).",
        "space_complexity": "O(k) for storing the heap, where k is the number of elements to be tracked.",
        "detailed_approach": [
            "1. Initialize Heap with Comparator: Create a min-heap or max-heap based on the problem's requirement (e.g., min-heap for finding k largest elements). The comparator function determines the order of elements in the heap.",
            "2. Process Elements Maintaining Heap: Iterate through the input elements. For each element, insert it into the heap. If the heap size exceeds k, remove the root element (smallest in a min-heap, largest in a max-heap) to maintain the desired size.",
            "3. Extract Top Elements as Needed: After processing all elements, the heap will contain the k smallest or largest elements. Extract these elements as needed, typically by repeatedly removing the root.",
            "4. Update Heap Efficiently: When inserting or removing elements, ensure that the heap property is maintained. This involves heapify operations (up-heap or down-heap) to restore the correct order.",
            "Example Use Case: Finding the k largest numbers in a stream, merging k sorted lists, implementing a priority-based task scheduler, and solving problems involving finding medians or other order statistics.",
        ],
    },
    "k_sum": {
        "suggestion": "Apply the k-sum approach, often involving sorting and the two-pointer technique, to find combinations of k elements that satisfy a specific sum condition in an array.",
        "time_complexity": "O(n^(k-1)) for the general k-sum problem, where n is the length of the array. For example, 2-sum is O(n) after sorting, 3-sum is O(n²), and so on.",
        "space_complexity": "O(1) excluding the space required for the output. Sorting may require O(log n) space depending on the algorithm used.",
        "detailed_approach": [
            "1. Sort Array if Needed: If the input array is not already sorted, sort it first. This step is crucial for efficiently using the two-pointer technique in the subsequent steps.",
            "2. Fix k-2 Elements with Loops: Use nested loops to fix the first k-2 elements of the k-tuple. For example, in 3-sum, you would use one loop to fix the first element; in 4-sum, you would use two nested loops to fix the first two elements.",
            "3. Use Two Pointers for Remaining Sum: For the remaining two elements, use the two-pointer technique. Initialize one pointer at the position immediately after the last fixed element and the other at the end of the array. Move these pointers towards each other based on the sum of the elements they point to, compared to the target sum.",
            "4. Handle Duplicates Carefully: When moving pointers or fixing elements, ensure that duplicates are handled properly to avoid duplicate combinations in the output. This often involves skipping over elements that are identical to the previous one.",
            "5. Combine and Record Results: When a combination that sums up to the target is found, record it. Continue moving the pointers until they cross each other, then proceed with the next iteration of the outer loops.",
            "Example Use Case: Finding all unique triplets in an array that sum up to zero (3-sum), finding all unique quadruplets that sum up to a specific target (4-sum), and generalizing to higher k values for similar problems.",
        ],
    },
    "string_manipulation": {
        "suggestion": "Employ string manipulation techniques for problems that involve processing and transforming strings, such as finding patterns, reversing, or rearranging characters.",
        "time_complexity": "O(n) for a single pass through the string, where n is the length of the string. Operations like string concatenation or substring extraction can affect the complexity.",
        "space_complexity": "O(n) in the worst case, especially when creating new strings. In-place operations can reduce space usage.",
        "detailed_approach": [
            "1. Process Characters Systematically: Iterate through the string, often character by character, and apply the necessary operations based on the problem's requirements.",
            "2. Use Appropriate String Methods: Utilize built-in string methods for common tasks like finding substrings, replacing characters, splitting, or joining strings. Be aware of the performance implications of these methods.",
            "3. Handle Edge Cases and Boundaries: Pay attention to edge cases, such as empty strings, strings with a single character, or special characters. Ensure that your solution correctly handles these cases.",
            "4. Consider String Immutability: In many languages, strings are immutable. This means that operations that appear to modify a string actually create a new string. Be mindful of this when performing repeated modifications, as it can lead to performance issues.",
            "5. Use String Builders for Efficiency: When performing a large number of string concatenations or modifications, consider using a string builder or similar construct to improve efficiency.",
            "Example Use Case: Reversing a string, finding the first non-repeating character, checking for palindromes, validating parentheses, and implementing string compression or decompression algorithms.",
        ],
    },
    "tree_traversal": {
        "suggestion": "Employ tree traversal techniques (DFS: pre-order, in-order, post-order; BFS: level-order) for problems involving tree structures, especially when needing to visit all nodes in a specific order.",
        "time_complexity": "O(n) for traversing all nodes in the tree, where n is the number of nodes.",
        "space_complexity": "O(h) in the average case for recursive DFS, where h is the height of the tree (O(log n) for balanced trees, O(n) for skewed trees). O(w) for BFS, where w is the maximum width of the tree.",
        "detailed_approach": [
            "1. Choose Traversal Type: Select the appropriate traversal method based on the problem's requirements. DFS is useful for exploring paths and going deep into the tree, while BFS is suitable for finding the shortest path or processing nodes level by level.",
            "2. Implement Recursive/Iterative: Implement the chosen traversal method either recursively or iteratively. Recursive implementations are often more concise but can lead to stack overflow for very deep trees. Iterative implementations using a stack (for DFS) or queue (for BFS) can avoid this issue.",
            "3. Process Nodes During Traversal: Perform the necessary operations on each node as it is visited. The order of processing depends on the traversal type (e.g., pre-order processes the node before its children, in-order processes the node between its left and right subtrees, post-order processes the node after its children).",
            "4. Handle Null Nodes Properly: Ensure that your traversal algorithm correctly handles null nodes, which represent the absence of a child node. This is typically done by checking for null before making recursive calls or adding nodes to the stack/queue.",
            "Example Use Case: Searching for a node with a specific value, calculating the height or depth of a tree, performing tree serialization/deserialization, and solving problems that require visiting all nodes in a specific order (e.g., evaluating expressions in an expression tree).",
        ],
    },
    "hash_table": {
        "suggestion": "Utilize hash tables (dictionaries, maps) for problems that require efficient lookups, insertions, and deletions, especially when dealing with key-value pairs or checking for the existence of elements.",
        "time_complexity": "O(n) for building a hash table from n elements. O(1) on average for lookups, insertions, and deletions, but can degrade to O(n) in the worst case due to hash collisions.",
        "space_complexity": "O(n) for storing n elements in the hash table.",
        "detailed_approach": [
            "1. Choose Key-Value Structure: Determine the appropriate key-value structure for the problem. Keys should be unique and immutable, while values can be of any type.",
            "2. Handle Collisions if Needed: If using a language or implementation that doesn't handle collisions automatically, implement a collision resolution strategy (e.g., chaining, open addressing).",
            "3. Process Elements Efficiently: Use hash table operations (e.g., `get`, `put`, `contains`) to efficiently process elements. Leverage the fast lookup capability to check for the existence of elements, retrieve associated values, or update entries.",
            "4. Consider Load Factor: Be mindful of the hash table's load factor (ratio of elements to buckets). A high load factor can increase the likelihood of collisions and degrade performance. Some implementations automatically resize the hash table when the load factor exceeds a threshold.",
            "Example Use Case: Counting the frequency of elements in an array, implementing a cache, detecting duplicates, finding pairs of elements that satisfy a certain condition, and solving problems that require mapping keys to values.",
        ],
    },
    "stack": {
        "suggestion": "Employ stacks for problems that require Last-In-First-Out (LIFO) processing, such as evaluating expressions, managing function calls, or implementing undo/redo functionality.",
        "time_complexity": "O(n) for processing n elements using a stack, where each push and pop operation takes O(1) time.",
        "space_complexity": "O(n) in the worst case, where all elements are pushed onto the stack.",
        "detailed_approach": [
            "1. Initialize Stack Structure: Create an empty stack to store elements.",
            "2. Push/Pop Elements as Needed: Use the `push` operation to add elements to the top of the stack and the `pop` operation to remove elements from the top. Follow the LIFO principle to process elements in the correct order.",
            "3. Track Stack State: Keep track of the current state of the stack, such as the top element or the number of elements, to make decisions during processing.",
            "4. Handle Empty Stack Cases: Ensure that your solution correctly handles cases where the stack is empty, such as when trying to pop from an empty stack.",
            "Example Use Case: Evaluating arithmetic expressions, parsing programming language syntax, implementing browser history, managing function calls in a program, and solving problems that require reversing the order of elements or tracking a sequence of actions.",
        ],
    },
    "greedy": {
        "suggestion": "Apply a greedy approach for optimization problems where making locally optimal choices at each step can lead to a globally optimal solution. This often involves sorting or prioritizing elements based on some criteria.",
        "time_complexity": "Often O(n log n) due to sorting, but can be O(n) if sorting is not required or if a priority queue is used efficiently.",
        "space_complexity": "O(1) if the greedy algorithm can be implemented in-place, or O(n) if additional data structures are needed to store intermediate results.",
        "detailed_approach": [
            "1. Sort if Beneficial: If the problem involves selecting elements in a specific order, sort the input based on the criteria that define the locally optimal choice.",
            "2. Make Locally Optimal Choices: At each step, select the element or make the decision that appears to be the best at that moment, without considering future consequences.",
            "3. Prove Global Optimality (if possible): For a greedy algorithm to be correct, it must be proven that the locally optimal choices always lead to a globally optimal solution. This often involves an inductive argument or a proof by contradiction.",
            "4. Handle Edge Cases: Ensure that the greedy strategy correctly handles edge cases and produces the correct result for all valid inputs.",
            "Example Use Case: Activity selection problem, Huffman coding, Kruskal's algorithm for minimum spanning trees, Dijkstra's algorithm for shortest paths in graphs with non-negative edge weights, and coin change problem (for certain coin systems).",
        ],
    },
    "union_find": {
        "suggestion": "Employ the Union-Find (Disjoint Set) data structure for problems involving dynamic connectivity, such as determining if two elements belong to the same set, merging sets, or tracking connected components in a graph.",
        "time_complexity": "O(α(n)) amortized time per operation (find or union), where α(n) is the inverse Ackermann function, which grows very slowly and is effectively constant for all practical values of n.",
        "space_complexity": "O(n) for storing the parent array and, optionally, the rank or size array.",
        "detailed_approach": [
            "1. Initialize Parent Array: Create an array `parent` of size n, where `parent[i]` represents the parent of element i. Initially, each element is its own parent (i.e., `parent[i] = i`).",
            "2. Implement Find with Path Compression: The `find` operation determines the root (representative) of the set to which an element belongs. Implement `find` recursively, and during each call, update the parent of the current element to the root (path compression). This optimization flattens the tree structure and improves the efficiency of subsequent `find` operations.",
            "3. Union by Rank or Size: The `union` operation merges two sets. To keep the tree structure relatively flat, use union by rank or size. In union by rank, the root of the shorter tree becomes the child of the root of the taller tree. In union by size, the root of the smaller tree becomes the child of the root of the larger tree.",
            "4. Process Connections: Use `find` and `union` to process connections between elements. To check if two elements are in the same set, compare their roots (`find(a) == find(b)`). To merge two sets, perform `union(find(a), find(b))`. ",
            "Example Use Case: Kruskal's algorithm for minimum spanning trees, detecting cycles in a graph, solving network connectivity problems, and implementing social network analysis algorithms.",
        ],
    },
    "bit_manipulation": {
        "suggestion": "Utilize bit manipulation techniques for problems that involve operating on individual bits of integers, such as optimizing space, performing fast calculations, or representing sets.",
        "time_complexity": "O(1) for basic bit operations (AND, OR, XOR, NOT, shifts). O(log n) for operations that depend on the number of bits in an integer.",
        "space_complexity": "O(1) - bit manipulation is typically performed in-place.",
        "detailed_approach": [
            "1. Identify Bit Operations Needed: Determine the specific bit operations required for the problem, such as setting, clearing, toggling, or checking bits.",
            "2. Use Masks and Shifts: Use bit masks (integers with specific bits set) to isolate or modify specific bits. Use left and right shifts to move bits to the desired positions.",
            "3. Handle Edge Cases: Be mindful of edge cases, such as negative numbers (which are typically represented using two's complement) and overflow.",
            "4. Consider Bit Properties: Leverage properties of bit operations, such as XORing a number with itself to get 0, or using AND with a power of 2 to check if a number is a power of 2.",
            "Example Use Case: Swapping two numbers without a temporary variable, checking if a number is even or odd, finding the single number that appears only once in an array where all other numbers appear twice, implementing bitsets, and optimizing algorithms that involve subsets or power sets.",
        ],
    },
    "prefix_sum": {
        "suggestion": "Employ prefix sums for problems that involve repeatedly calculating the sum of elements in a range of an array, especially when multiple queries are performed on the same array.",
        "time_complexity": "O(n) for building the prefix sum array. O(1) for each range sum query.",
        "space_complexity": "O(n) for storing the prefix sum array.",
        "detailed_approach": [
            "1. Build Prefix Sum Array: Create an array `prefix_sum` of the same size as the input array. Calculate the prefix sums iteratively: `prefix_sum[0] = arr[0]`, and `prefix_sum[i] = prefix_sum[i-1] + arr[i]` for i > 0.",
            "2. Handle Range Queries: To find the sum of elements in the range [left, right], use the formula `sum[left, right] = prefix_sum[right] - prefix_sum[left-1]` (handle the case where `left = 0` separately).",
            "3. Consider Cumulative Properties: When applicable, leverage the cumulative properties of prefix sums to efficiently calculate other range-based statistics, such as averages or differences.",
            "4. Watch for Overflow: Be mindful of potential integer overflow when calculating prefix sums, especially if the input array contains large numbers.",
            "Example Use Case: Finding the sum of elements in a given range of an array, calculating moving averages, solving problems that involve subarray sums, and implementing 2D prefix sums for matrix range queries.",
        ],
    },
    "intervals": {
        "suggestion": "Employ interval-based techniques for problems that involve processing intervals or ranges, such as merging overlapping intervals, finding intersections, or scheduling events.",
        "time_complexity": "O(n log n) for sorting the intervals, where n is the number of intervals. O(n) for merging or processing intervals after sorting.",
        "space_complexity": "O(n) in the worst case, such as when all intervals are disjoint and need to be stored in the result.",
        "detailed_approach": [
            "1. Sort Intervals Appropriately: If the intervals are not already sorted, sort them based on their start times (or end times, depending on the problem).",
            "2. Process in Order: Iterate through the sorted intervals, keeping track of the current interval or merged interval.",
            "3. Handle Overlaps: When processing intervals, check for overlaps with the current interval. If an overlap is found, merge the intervals or update the current interval accordingly.",
            "4. Track Boundaries: Keep track of the start and end times of the current interval or merged interval to efficiently handle subsequent intervals.",
            "Example Use Case: Merging overlapping intervals, finding the intersection of intervals, inserting a new interval into a set of non-overlapping intervals, scheduling meetings, and solving problems that involve time ranges or resource allocation.",
        ],
    },
    "matrix": {
        "suggestion": "Employ matrix traversal techniques for problems that involve 2D grids or matrices, such as searching for elements, finding paths, or performing operations on rows and columns.",
        "time_complexity": "O(mn) for traversing all elements in an m x n matrix.",
        "space_complexity": "O(1) for in-place operations or O(mn) for storing a copy of the matrix or intermediate results.",
        "detailed_approach": [
            "1. Choose Traversal Pattern: Select an appropriate traversal pattern based on the problem's requirements. Common patterns include row-major order, column-major order, diagonal traversal, and spiral traversal.",
            "2. Handle Boundaries: Ensure that your traversal algorithm correctly handles the boundaries of the matrix to avoid going out of bounds.",
            "3. Process Elements Systematically: Perform the necessary operations on each element as it is visited. This could involve checking properties, updating values, or comparing with neighboring elements.",
            "4. Consider Direction Arrays: For problems that involve moving in multiple directions (e.g., up, down, left, right), use direction arrays to simplify the code and make it more readable.",
            "Example Use Case: Searching for a specific value in a matrix, finding the shortest path in a maze, rotating a matrix, performing matrix operations (e.g., addition, multiplication), and solving problems that involve game boards or grids.",
        ],
    },
    "monotonic_stack": {
        "suggestion": "Employ a monotonic stack (increasing or decreasing) for problems that involve finding the next greater element, previous smaller element, or maintaining a specific order of elements.",
        "time_complexity": "O(n) amortized, as each element is pushed onto and popped from the stack at most once.",
        "space_complexity": "O(n) in the worst case, where all elements are pushed onto the stack.",
        "detailed_approach": [
            "1. Maintain Monotonic Property: Use a stack to store elements in increasing or decreasing order. When processing a new element, compare it with the top of the stack. If the new element violates the monotonic property, pop elements from the stack until the property is restored or the stack is empty.",
            "2. Process Elements in Order: Iterate through the input elements in the given order. For each element, perform the necessary operations based on the problem's requirements.",
            "3. Handle Stack Operations: When popping elements from the stack, perform the required calculations or updates. This often involves using the popped element and the current element to determine a result or update an array.",
            "4. Build Result Array: After processing all elements, the stack may contain elements that do not have a next greater element (or previous smaller element). Handle these elements appropriately, often by setting their result to a default value.",
            "Example Use Case: Finding the next greater element for each element in an array, calculating the maximum area of a histogram, finding the largest rectangle in a binary matrix, and solving problems that involve stock spans or similar patterns.",
        ],
    },
    "trie": {
        "suggestion": "Employ a trie (prefix tree) for problems that involve efficient string retrieval, prefix matching, or storing a set of strings.",
        "time_complexity": "O(L) for inserting a string of length L into the trie. O(L) for searching for a string of length L. O(L) for prefix-based operations, where L is the length of the prefix.",
        "space_complexity": "O(NL) in the worst case, where N is the number of strings and L is the average length of the strings. However, the space complexity can be lower in practice if the strings share common prefixes.",
        "detailed_approach": [
            "1. Design Trie Structure: Create a trie node structure that contains an array of children (one for each possible character) and a flag to indicate whether the node represents the end of a valid word.",
            "2. Implement Insert/Search: Implement the `insert` operation to add a string to the trie. Traverse the trie, creating new nodes as needed, and mark the final node as the end of a word. Implement the `search` operation to check if a string exists in the trie. Traverse the trie based on the characters of the string, and return true if the final node is marked as the end of a word.",
            "3. Track Word Endings: Use a boolean flag or a special marker in each node to indicate whether the path to that node represents a complete word.",
            "4. Optimize Space Usage: Consider using techniques like compressed tries or ternary search trees to reduce the space usage of the trie, especially when dealing with a large number of strings.",
            "Example Use Case: Implementing autocomplete or spell-checking features, storing a dictionary of words, solving problems that involve finding all words that start with a given prefix, and implementing IP routing algorithms.",
        ],
    },
    "divide_and_conquer": {
        "suggestion": "Employ the divide and conquer strategy for problems that can be broken down into smaller, independent subproblems, solved recursively, and then combined to obtain the final solution.",
        "time_complexity": "Often O(n log n), such as in merge sort and quicksort. However, the time complexity depends on the specific problem and the recurrence relation.",
        "space_complexity": "O(log n) for the recursion stack in balanced cases, such as in merge sort. O(n) in the worst case for unbalanced recursion, such as in quicksort with a poor pivot choice.",
        "detailed_approach": [
            "1. Divide Problem Logically: Break down the problem into smaller, independent subproblems of the same type.",
            "2. Solve Subproblems Recursively: Solve each subproblem recursively by applying the same divide and conquer strategy. This step continues until the base cases are reached.",
            "3. Combine Results: Combine the solutions of the subproblems to obtain the solution to the original problem.",
            "4. Handle Base Cases: Define the base cases for the recursion, which are the smallest instances of the problem that can be solved directly without further division.",
            "Example Use Case: Merge sort, quicksort, binary search, finding the closest pair of points, Strassen's matrix multiplication algorithm, and solving the Tower of Hanoi puzzle.",
        ],
    },
    "recursion": {
        "suggestion": "Employ recursion for problems that exhibit a self-similar structure, where the solution can be expressed in terms of solutions to smaller instances of the same problem.",
        "time_complexity": "Varies depending on the problem and the number of recursive calls. Can range from O(log n) for problems like binary search to O(2^n) for problems with exponential branching, such as calculating Fibonacci numbers without memoization.",
        "space_complexity": "O(h) for the recursion stack, where h is the maximum depth of the recursion tree. In the worst case, h can be equal to n, leading to O(n) space complexity.",
        "detailed_approach": [
            "1. Define Base Cases: Identify the base cases, which are the simplest instances of the problem that can be solved directly without further recursion.",
            "2. Implement Recursive Step: Express the solution to the problem in terms of solutions to smaller instances of the same problem. Make recursive calls to solve these smaller instances.",
            "3. Ensure Progress: Ensure that each recursive call makes progress towards a base case. This is crucial to prevent infinite recursion.",
            "4. Consider Stack Depth: Be mindful of the maximum depth of the recursion, as each recursive call adds a new frame to the call stack. Deep recursion can lead to stack overflow errors.",
            "Example Use Case: Calculating factorials, traversing tree or graph structures, implementing backtracking algorithms, solving problems with recursive definitions (e.g., Fibonacci numbers, Tower of Hanoi), and implementing divide and conquer algorithms.",
        ],
    },
    "bitmask": {
        "suggestion": "Employ bitmasks for problems that involve representing sets or subsets, performing operations on multiple elements simultaneously, or optimizing space when dealing with a small number of items.",
        "time_complexity": "O(2^n) for iterating through all possible subsets of n items. O(1) for basic bitwise operations.",
        "space_complexity": "O(1) per state when using an integer to represent a state. O(2^n) for storing all possible states.",
        "detailed_approach": [
            "1. Design State Representation: Use an integer as a bitmask, where each bit represents the presence or absence of an element in a set or a specific state.",
            "2. Implement Bit Operations: Use bitwise operations (AND, OR, XOR, shifts) to manipulate the bitmask. For example, set a bit to 1 to add an element to the set, clear a bit to 0 to remove an element, or use XOR to toggle the presence of an element.",
            "3. Track State Changes: Update the bitmask based on the problem's requirements. This often involves iterating through all possible subsets or performing transitions between states.",
            "4. Handle Transitions: Define how the bitmask changes from one state to another. This may involve adding or removing elements, combining states, or applying specific rules based on the problem.",
            "Example Use Case: Solving problems that involve subsets or power sets, such as finding all possible combinations of items, implementing bitset data structures, optimizing dynamic programming algorithms that involve subsets, and solving problems where the state can be compactly represented using bits.",
        ],
    },
    "binary_tree": {
        "suggestion": "Employ binary tree data structures for hierarchical data representation and efficient searching, insertion, and deletion operations.",
        "time_complexity": "O(n) for traversal operations where n is the number of nodes. O(log n) for balanced BST operations like search/insert/delete.",
        "space_complexity": "O(h) where h is the height of the tree. O(log n) for balanced trees, O(n) for skewed trees.",
        "detailed_approach": [
            "1. Choose Traversal Method: Select appropriate traversal (inorder, preorder, postorder, level-order) based on problem requirements. Each traversal visits nodes in a specific order useful for different scenarios.",
            "2. Handle Tree Properties: Utilize binary tree properties like parent-child relationships, BST ordering, height-balance conditions. Consider special cases like leaf nodes and single-child nodes.",
            "3. Process Nodes Recursively: Implement recursive solutions leveraging the tree's natural recursive structure. Use base cases for leaf nodes or null nodes. Consider iterative approaches with stacks/queues for better space complexity.",
            "4. Maintain Balance: For BSTs, consider self-balancing mechanisms like AVL or Red-Black trees to ensure O(log n) operations. Track height/balance factors during modifications.",
            "Example Use Case: Implementing hierarchical data structures, binary search trees for efficient searching, expression trees for parsing, and Huffman coding trees for compression.",
        ],
    },
    "topological_sort": {
        "suggestion": "Apply topological sorting for problems involving directed acyclic graphs (DAGs) where tasks/items need to be ordered based on their dependencies.",
        "time_complexity": "O(V + E) where V is the number of vertices and E is the number of edges in the graph.",
        "space_complexity": "O(V) for storing the visited set and result array. Additional O(V) may be needed for queue/stack in implementation.",
        "detailed_approach": [
            "1. Build Adjacency Structure: Create an adjacency list or matrix representation of the graph. For each vertex, maintain a list of its outgoing edges or dependencies.",
            "2. Track Node States: Maintain visited sets and in-degree counts for each vertex. This helps detect cycles and determine which nodes are ready for processing.",
            "3. Process Nodes Systematically: Use either DFS or BFS approach. For Kahn's algorithm, start with nodes having zero in-degree and progressively remove edges while adding nodes to result.",
            "4. Handle Cycles: Implement cycle detection as topological sort is only valid for DAGs. Keep track of nodes in the current DFS path or check if all nodes are processed.",
            "Example Use Case: Course scheduling with prerequisites, build system dependency resolution, task scheduling with dependencies, and determining valid ordering in any system with dependencies.",
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
