from matplotlib.pyplot import uninstall_repl_displayhook
import requests
import unittest

class TestInference(unittest.TestCase):

    ENDPOINT = "http://localhost:8100"

    def test_inference(self):
        route = TestInference.ENDPOINT + "/model/classify_unlabel_ads"
        res = requests.get(route)
        self.assertEqual(res.status_code, 200)

if __name__ == "__main__":
    unittest.main()
