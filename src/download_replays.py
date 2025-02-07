import requests
import os


# Get the replay group id
url = 'https://ballchasing.com/api/groups?creator=76561198156732134'
token = 'S3DBRnFgr6SMYDtu9FVgP5gqe6gB1MECRlKjrQqT'
headers = {
    'Authorization': f'{token}',
    'Content-Type': 'application/json'
}
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print(response.status_code)
    print(data)
else:
    print(f"Failed to retrieve data: {response.status_code}")


# Adding replay ids into an array of ids
replays = []

url = 'https://ballchasing.com/api/replays?group=set5-4bw2ycangs&count=200'
headers = {
    'Authorization': f'{token}',
    'Content-Type': 'application/json'
}
response = requests.get(url, headers=headers)
if response.status_code == 200:
    data = response.json()
    for replay_id in data["list"]:
        replays.append(replay_id["id"])
else:
    print(f"Failed to retrieve data: {response.status_code}")

# Downloads replay files from ballchasing and writes them to the replay files folder
iteration = 0
for replay in replays:
    iteration += 1
    url = f'https://ballchasing.com/api/replays/{replay}/file'
    token = 'S3DBRnFgr6SMYDtu9FVgP5gqe6gB1MECRlKjrQqT'
    headers = {
        'Authorization': f'{token}',
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': 'attachment; filename="original-filename.replay"'
    }
    response = requests.get(url, headers=headers)
    folder_path = 'C:\\Users\\Marco\\Desktop\\Python\\data challenges\\rocket car\\replay files'
    replay_path = os.path.join(folder_path, f'{replay}.replay')

    print(f"Iteration: {iteration}")
    if response.status_code == 200:
        print(f"Successfully retrieved data: {response.status_code}")
        with open(replay_path, 'wb') as file:
            file.write(response.content)
    elif response.status_code ==  429:
        print("You have hit the rate limit. Cool down client and start again.")
        replays = [r for r in replays if r not in replays[0:iteration]]
        break
    else:
        print(f"Failed to retrieve data: {response.status_code}")