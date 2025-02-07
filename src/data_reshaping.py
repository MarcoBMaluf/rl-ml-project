import pandas as pd
import numpy as np
import sys
sys.path.append('C:\\...\\data tidying scripts')


player_data = pd.read_csv('C:\\...\\player_data.csv')
ball_data = pd.read_csv('C:\\...\\ball_data.csv')


from feature_engineering_functions import add_zone, distance_to_ball, closest_player_hitter

ball_data = add_zone(ball_data=ball_data)
player_data = distance_to_ball(match_data=player_data, ball_data=ball_data)
ball_data = closest_player_hitter(ball_data=ball_data, match_data=player_data)


cols_of_interest = [
'player',
'team',
'pos_x',
'pos_y',
'pos_z',
'vel_x',
'vel_y',
'vel_z',
'ang_vel_x',
'ang_vel_y',
'ang_vel_z',
'throttle',
'steer',
'handbrake',
'rot_x',
'rot_y',
'rot_z',
'double_jump_active',
'dodge_active',
'boost_active',
'jump_active',
'distance_to_ball'
]

for match in ball_data.match.unique():
    print(f'match: {match}')

    # isolate match from players and ball
    playersx = player_data[player_data['match'] == match]
    ballx = ball_data[ball_data['match'] == match].reset_index(drop=True)

    # get players
    players = playersx.player.unique()

    # take columns of interest
    playersx = playersx[cols_of_interest]

    # store each player data in a list
    playersx_individual_players = [playersx[playersx['player'] == player].reset_index(drop=True) for player in players]

    # rename the columns
    for i, player in enumerate(playersx_individual_players):

        # Generate new column names with player index using f-strings
        new_columns = {col: f'{col}_player{i+1}' for col in player.columns}
        
        # Rename columns inplace
        player.rename(columns=new_columns, inplace=True) # type: ignore
    
    # join the players together
    playersx_joined = pd.concat(playersx_individual_players, axis=1)

    # join players with ball
    matchx = pd.concat([ballx, playersx_joined], axis=1)

    # make singular dataframe
    if match > 1:
        match_data = pd.concat([match_data, matchx], axis=0, ignore_index=True)
    else:
        match_data = matchx
    print('---------')

# verifying the shape of the data
match_data.shape
ball_data.shape


match_data.to_csv('match_data.csv', index=False)