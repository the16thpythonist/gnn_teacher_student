#!/usr/bin/env python
"""Tests for `gnn_teacher_student` package."""
import random

import unittest

from gnn_teacher_student.main import StudentTeacherExplanationAnalysis


class TestSnippets(unittest.TestCase):

    def test_random_color(self):
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        weights = [1, 1, 1]

        # random.choices with an "s" always returns a list regardless of k=1 or otherwise!
        result = random.choices(colors, weights=weights, k=1)
        self.assertTrue(isinstance(result, list))
        self.assertEqual(1, len(result))
        self.assertTrue(isinstance(result[0], tuple))


class StudentTeacherExplanationAnalysisTest(unittest.TestCase):

    def test_construction_basically_works(self):
        student_teacher_analysis = StudentTeacherExplanationAnalysis(
            student_models={},
            loss=[]
        )

        self.assertTrue(isinstance(student_teacher_analysis, StudentTeacherExplanationAnalysis))
