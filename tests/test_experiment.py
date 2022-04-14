import os
import unittest
import tempfile

from gnn_teacher_student.experiment import Experiment


class TestExperiment(unittest.TestCase):

    def test_construction_basically_works(self):

        with tempfile.TemporaryDirectory() as base_path:
            with Experiment(base_path, 'test_experiment', '') as e:
                self.assertTrue(os.path.exists(e.path))
                self.assertTrue(os.path.isdir(e.path))

    def test_override_flag_works(self):

        with tempfile.TemporaryDirectory() as base_path:
            # Here we set up an initial folder with some contents, this will be the initial existing
            # folder to test the overriding in the next step
            with Experiment(base_path, 'test_experiment', '') as e:
                file_path = os.path.join(e.path, 'hello_world.txt')
                with open(file_path, mode='w') as file:
                    file.write('HELLO WORLD!')

                self.assertTrue(os.path.exists(file_path))
                self.assertTrue(os.path.isfile(file_path))

            # First we will test if the folder is NOT deleted if the override flag is false and instead
            # this second experiment sets up a different folder
            with Experiment(base_path, 'test_experiment', '', override=False) as a:
                self.assertTrue(os.path.exists(a.path))
                self.assertTrue(os.path.exists(e.path))
                self.assertTrue(os.path.exists(file_path))
                self.assertNotEqual(e.path, a.path)

            # Now we setup an experiment which should override the first one. We will see if that worked
            # by checking if the file we created initially is gone after the override process
            with Experiment(base_path, 'test_experiment', '', override=True) as b:
                self.assertTrue(os.path.exists(b.path))
                self.assertEqual(e.path, b.path)
                self.assertFalse(os.path.exists(file_path))

    def test_artifacts_are_created(self):

        with tempfile.TemporaryDirectory() as base_path:

            description = 'This is a test description'
            with Experiment(base_path, 'test_experiment', description) as e:
                # The description file is supposed to be created on enter
                self.assertTrue(os.path.exists(e.description_path))
                with open(e.description_path) as file:
                    content = file.read()
                    self.assertEqual(content, description)

                # The log file should also already exist from the beginning
                self.assertTrue(os.path.exists(e.log_path))
                self.assertTrue(os.path.isfile(e.log_path))

                # The report file is created on exit, which is why it should not exist within the content
                # yet, only after
                self.assertFalse(os.path.exists(e.report_path))

            self.assertTrue(os.path.exists(e.report_path))

    def test_exceptions_are_correctly_handled(self):

        with tempfile.TemporaryDirectory() as base_path:

            with Experiment(base_path, 'test_experiment', '') as e:

                # This exception should be handled by the context and not disrupt the program
                raise Exception('some error...')

            # If an exception occurs within the experiment context, an error log file should be created!
            self.assertTrue(os.path.exists(e.error_path))

            # If No exception occurs, the error report should NOT be created however
            with Experiment(base_path, 'test_experiment', '', override=True) as e:
                pass

            self.assertFalse(os.path.exists(e.error_path))
