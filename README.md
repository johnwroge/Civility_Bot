# Discord_Detection_Bot

## Testing

python bot.py

Running the Tests
To run all tests:
bashpytest -xvs test_bot.py
To run a specific test class:
bashpytest -xvs test_bot.py::TestHateSpeechDetector
To run a specific test:
bashpytest -xvs test_bot.py::TestHateSpeechDetector::test_predict_hate_speech