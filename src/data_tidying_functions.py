import carball
from carball.json_parser.game import Game
from carball.analysis.analysis_manager import AnalysisManager
import json
import pandas as pd
import chardet
import warnings
warnings.filterwarnings('ignore')

'''
make_data_frame() function:
'''

def make_dataframe(json_file_path):

    '''
    This functions takes a .json file and converts it into a pandas DataFrame through
    the use of carball's get_data_frame() function from the analysis_manager class.
    Returns a DataFrame called main_DataFrame
    '''

    with open(json_file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']

    with open(json_file_path, encoding=encoding) as f:
        _json = json.load(f)

    # _json is a JSON game object
    game = Game()
    game.initialize(loaded_json=_json)

    analysis_manager = AnalysisManager(game)
    analysis_manager.create_analysis()

    # return the pandas data frame in python
    main_DataFrame = analysis_manager.get_data_frame()

    return main_DataFrame


'''
tidy_match_data() function:
'''

def tidy_match_data(df, match_num):

    '''
    Makes a singular DataFrame from the original one made in make_dataframe().

    Takes the player-based data from main_DataFrame and tidies them into one DataFrame.

    Adds a player column recording the player the proceeding columns describe, time-based 
    columns from main_DataFrame['game'], a team column that records the team that player is on, 
    and a match column that records the specific match the data is from.
    
    The singular DataFrame that is made is to be merged with all the other matches
    and their data within the script.
    '''

    # get a list of unique player names:
    cols=[]
    i=0
    while i<len(df.columns):
        cols.append(df.columns[i][0])
        i+=1
    players = list(set(cols))
    players.remove('game')
    players.remove('ball')

    # add each player's DataFrame into a list of DataFrames
    player_data_frames = [df[name] for name in players]

    # add player column to individual player's data:
    i=0
    for player_data_frame in player_data_frames:
        player_data_frame.insert(0, 'player', players[i])
        i+=1

    # add team column to individual player's data:
    for player_data_frame in player_data_frames:
        if player_data_frame.iloc[0].pos_y < 0:
            player_data_frame.insert(1, 'team', 0)
        elif player_data_frame.iloc[0].pos_y > 0:
            player_data_frame.insert(1, 'team', 1)

    # add relevant columns to individual player's data from main_DataFrame['game']
    if 'is_overtime' in df['game'].columns:
        game_cols = [
        'time', 'seconds_remaining', 'is_overtime', 'ball_has_been_hit', 'goal_number'
        ]
    else: 
        game_cols = [
        'time', 'seconds_remaining', 'ball_has_been_hit', 'goal_number'
        ]
    for player_data_frame in player_data_frames:
        player_data_frame[game_cols] = df['game'][game_cols]

    # merge all individual data together into one DataFrame called player_data_temp
    player_data_temp = pd.concat(player_data_frames)

    # add match column
    player_data_temp['match'] = match_num

    return player_data_temp

'''
tidy_ball_data() function:
'''

def tidy_ball_data(df, match_num):

    '''
    Makes a singular DataFrame from the original one made in make_dataframe().

    Takes the ball data from main_DataFrame and tidies it into one DataFrame.
    
    Adds time column from main_DataFrame['game'] and
    a match column that records the specific match the data is from.

    The singular DataFrame that is made is to be merged with all the other matches
    and their data within the script.
    '''

    # store ball data
    ball_data_temp = df['ball']

    # add time column
    ball_data_temp['time'] = df['game']['time']

    # add match column
    ball_data_temp['match'] = match_num

    return ball_data_temp