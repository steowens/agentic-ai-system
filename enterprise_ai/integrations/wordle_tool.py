"""
Wordle constraint solver tool.
Provides algorithmic solutions for Wordle puzzles with letter constraints.
"""
from typing import Set, Dict
from autogen_core.tools import FunctionTool
from .base_tool import BaseTool


class WordleTool(BaseTool):
    """Wordle constraint solver tool"""

    def __init__(self):
        self._description = "Solves Wordle word puzzles with letter constraints, positions, and exclusions"

    def solve_wordle(self, excluded_letters: str, included_letters: str, letters_at: str) -> str:
        """
        Solve Wordle constraints to find valid words.

        Args:
            excluded_letters: Letters that cannot be in the word (e.g., "CORIU")
            included_letters: Letters that must be in the word, duplicates indicate required count (e.g., "PS" or "PP" for two P's)
            letters_at: Position constraints as "letter:position" pairs separated by commas (e.g., "P:2,S:4" means P at position 2, S at position 4)

        Returns:
            String with matching Wordle words or explanation if none found
        """

        # Parse the inputs
        try:
            excluded = set(excluded_letters.upper().replace(" ", ""))

            # Count required letters (handle duplicates)
            included_counts = {}
            for letter in included_letters.upper().replace(" ", ""):
                included_counts[letter] = included_counts.get(letter, 0) + 1

            # Parse position constraints
            position_constraints = {}
            if letters_at.strip():
                for constraint in letters_at.split(","):
                    if ":" in constraint:
                        letter, pos = constraint.strip().split(":")
                        position_constraints[int(pos)] = letter.upper()

            # Call your algorithm implementation
            return self._solve_wordle_constraints(excluded, included_counts, position_constraints)

        except Exception as e:
            return f"Error parsing Wordle constraints: {str(e)}"

    def _solve_wordle_constraints(self, excluded: Set[str], included_counts: Dict[str, int], position_constraints: Dict[int, str]) -> str:
        """
        YOUR ALGORITHM IMPLEMENTATION GOES HERE

        Args:
            excluded: set of letters to exclude (e.g., {'C', 'O', 'R', 'T', 'I', 'U'})
            included_counts: required letters and counts (e.g., {'P': 1, 'S': 1})
            position_constraints: position requirements (e.g., {2: 'P'} means P at position 2)

        Returns:
            String with found words or "No words found" message
        """
        import os
        from collections import Counter

        # Load word list from file
        words_file = os.getenv("WORDLE_WORDS_FILE_PATH", "data/valid-wordle-words.txt")

        try:
            with open(words_file, 'r') as f:
                all_words = [word.strip().upper() for word in f.readlines() if len(word.strip()) == 5]
        except FileNotFoundError:
            return f"Error: Word file '{words_file}' not found. Check WORDLE_WORDS_FILE_PATH environment variable."

        matching_words = []

        for word in all_words:
            # Check excluded letters
            if any(letter in word for letter in excluded):
                continue

            # Check position constraints
            position_match = True
            for pos, required_letter in position_constraints.items():
                if pos < 1 or pos > 5:  # Invalid position
                    continue
                if word[pos - 1] != required_letter:  # Convert to 0-based index
                    position_match = False
                    break

            if not position_match:
                continue

            # Check included letters and counts
            word_counter = Counter(word)
            included_match = True

            for letter, required_count in included_counts.items():
                if word_counter.get(letter, 0) < required_count:
                    included_match = False
                    break

            if included_match:
                matching_words.append(word)

        # Format results
        if matching_words:
            result = f"Found {len(matching_words)} matching words:\n"
            result += ", ".join(matching_words[:20])  # Show first 20 matches
            if len(matching_words) > 20:
                result += f"\n... and {len(matching_words) - 20} more"
        else:
            result = f"""No words found matching the constraints:
- Excluded letters: {excluded}
- Required letters: {included_counts}
- Position constraints: {position_constraints}

Try relaxing some constraints."""

        return result

    def get_function_tool(self) -> FunctionTool:
        return FunctionTool(self.solve_wordle, description=self._description)

    def get_description(self) -> str:
        return self._description