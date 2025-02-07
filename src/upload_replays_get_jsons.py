from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

# Specifying .replay files directory
replay_folder = 'C:\\...\\replay files'

# Set up Chrome options to handle downloads
download_dir = 'C:\\...\\replay .jsons' 
chrome_options = webdriver.ChromeOptions()
prefs = {'download.default_directory': download_dir}
chrome_options.add_experimental_option('prefs', prefs)

# Set up ChromeDriver Service
service = Service(ChromeDriverManager().install())

# Open Chrome and the site
driver = webdriver.Chrome(service=service, options= chrome_options)
driver.get('https://rl.nickb.dev/')

iteration = 0
for file in os.listdir(replay_folder):
    iteration += 1
    print(f"File Number: {iteration}")
    # Specify the replay's file path
    file_path = os.path.join(replay_folder, file)

    # Upload the .replay file
    wait = WebDriverWait(driver, 1)
    file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="file"]')))
    file_input.send_keys(file_path)

    # Wait a second to download the JSON
    time.sleep(1)

    # Download JSON file of replay
    json_download_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Convert Replay to JSON")]')))
    json_download_button.click()

    # Wait a second before the next iteration
    time.sleep(0.5)
