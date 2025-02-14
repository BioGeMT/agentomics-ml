import unittest
import subprocess
import os


class FileOperationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current_path = os.getcwd()
        cls.workspace_path = "/workspace"
        cls.test_file = "tests/blank_test_file.txt"
        cls.test_script = "tests/blank_test_script.py"
        os.makedirs('/workspace/tests', exist_ok=True)
    
    def write(self, file_path, content):
        try:
            with open(file_path, "w") as file:
                file.write(content)
                return True
        except:
            return False
    
    def read(self, file_path):
        try:
            with open(file_path, "r") as file:
                content = file.read()
                return True
        except:
            return False
    
    def delete(self, file_path):
        try:
            os.remove(file_path)
            return True
        except:
            return False
        
    def console_execute(self, file_path, check=True):
        try:
            subprocess.run(
                ["python", file_path], 
                check=check,
                stdout=subprocess.DEVNULL,  # suppress stdout
                stderr=subprocess.DEVNULL   # suppress stderr
            )
            return True
        except:
            return False
        
    def console_write(self, file_path, content, check=True):
        try:
            subprocess.run(
                ["touch", file_path], 
                check=check,
                stdout=subprocess.DEVNULL,  # suppress stdout
                stderr=subprocess.DEVNULL   # suppress stderr
            )
            subprocess.run(
                ["bash", "-c", f"echo '{content}' > {file_path}"], 
                check=check,
                stdout=subprocess.DEVNULL,  # suppress stdout
                stderr=subprocess.DEVNULL   # suppress stderr
            )
            return True
        except:
            return False
    
    def console_read(self, file_path, check=True):
        try:
            subprocess.run(
                ["cat", file_path], 
                check=check,
                stdout=subprocess.DEVNULL,  # suppress stdout
                stderr=subprocess.DEVNULL   # suppress stderr
            )
            return True
        except:
            return False
    
    def console_delete(self, file_path, check=True):
        try:
            subprocess.run(
                ["rm", "-f", file_path], 
                check=check,
                stdout=subprocess.DEVNULL,  # suppress stdout
                stderr=subprocess.DEVNULL   # suppress stderr
            )
            return True
        except:
            return False

    def test_read_file_inrepo(self):
        file_path = os.path.join(self.current_path, self.test_file)
        self.assertTrue(self.read(file_path))

    def test_write_file_inrepo(self):
        file_path = os.path.join(self.current_path, self.test_file)
        self.assertFalse(self.write(file_path, "This file has been written in."))

    def test_delete_file_inrepo(self):
        file_path = os.path.join(self.current_path, self.test_file)
        self.assertFalse(self.delete(file_path))

    def test_console_read_file_inrepo(self):
        file_path = os.path.join(self.current_path, self.test_file)
        self.assertTrue(self.console_read(file_path))

    def test_console_write_file_inrepo(self):
        file_path = os.path.join(self.current_path, self.test_file)
        self.assertFalse(
            self.console_write(
                file_path, 
                "This file has been written in using console commands.",
                check=True,
            )
        )

    def test_console_execute_inrepo(self):
        file_path = os.path.join(self.current_path, self.test_script)
        self.assertTrue(self.console_execute(file_path))

    def test_console_delete_file_inrepo(self):
        file_path = os.path.join(self.current_path, self.test_file)
        self.assertFalse(self.console_delete(file_path, check=True))

    def test_read_file_inworkspace(self):
        self.write(os.path.join(self.workspace_path, self.test_file),"")

        file_path = os.path.join(self.workspace_path, self.test_file)
        self.assertTrue(self.read(file_path))

    def test_write_file_inworkspace(self):
        file_path = os.path.join(self.workspace_path, self.test_file)
        self.assertTrue(self.write(file_path, "This file has been written in."))

    def test_delete_file_inworkspace(self):
        file_path = os.path.join(self.workspace_path, self.test_file)
        self.assertTrue(self.delete(file_path))

    def test_console_read_file_inworkspace(self):
        self.write(os.path.join(self.workspace_path, self.test_file),"")

        file_path = os.path.join(self.workspace_path, self.test_file)
        self.assertTrue(self.console_read(file_path))

    def test_console_write_file_inworkspace(self):
        file_path = os.path.join(self.workspace_path, self.test_file)
        self.assertTrue(self.console_write(file_path, "This file has been written in using console commands."))

    def test_console_delete_file_inworkspace(self):
        self.write(os.path.join(self.workspace_path, self.test_file),"")

        file_path = os.path.join(self.workspace_path, self.test_file)
        self.assertTrue(self.console_delete(file_path))

    def test_console_execute_inworkspace(self):
        self.write(os.path.join(self.workspace_path, self.test_script),"")
        file_path = os.path.join(self.workspace_path, self.test_script)
        self.assertTrue(self.console_execute(file_path))

if __name__ == '__main__':
    unittest.main()