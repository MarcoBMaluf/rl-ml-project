import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

'''
Distance in R3 of players from the ball

WORK FOR FUTURE 3v3 and 1v1 GAME DATA:
* Generalize the distance function and the for loop (all the *4 stuff needs to be *6 for 3s, *1 for 1s, having 1 section or 6 sections) *
* Somehow make it so that the number of sections is equal to the number of players *
'''

def distance(match_length, ballx, bally, ballz, matchx, matchy, matchz):
    
    sec1 = np.sqrt((matchx.iloc[0:match_length] - ballx)**2 + (matchy.iloc[0:match_length] - bally)**2 + (matchz.iloc[0:match_length] - ballz)**2)
    sec2 = np.sqrt((matchx.iloc[match_length:match_length*2].reset_index(drop=True) - ballx)**2 + (matchy.iloc[match_length:match_length*2].reset_index(drop=True) - bally)**2 + (matchz.iloc[match_length:match_length*2].reset_index(drop=True) - ballz)**2)
    sec3 = np.sqrt((matchx.iloc[match_length*2:match_length*3].reset_index(drop=True) - ballx)**2 + (matchy.iloc[match_length*2:match_length*3].reset_index(drop=True) - bally)**2 + (matchz.iloc[match_length*2:match_length*3].reset_index(drop=True) - ballz)**2)
    sec4 = np.sqrt((matchx.iloc[match_length*3:match_length*4].reset_index(drop=True) - ballx)**2 + (matchy.iloc[match_length*3:match_length*4].reset_index(drop=True) - bally)**2 + (matchz.iloc[match_length*3:match_length*4].reset_index(drop=True) - ballz)**2)

    distances = pd.concat([sec1, sec2, sec3, sec4], axis=0, ignore_index=True)

    return distances

'''
Adding `zone` column to ball data
'''

def add_zone(ball_data):

    conditions = [
        ball_data.pos_y < (-6000 + 10000/3),
        ball_data.pos_y > (6000 - 10000/3)
]

    choices = [-1, 1]

    ball_data['zone'] = np.select(conditions, choices, default= 0)

    return ball_data

'''
Adding `distance_to_ball ` column to match data
'''

def distance_to_ball(match_data, ball_data):

    # Get the number of matches
    number_of_matches = match_data.match.unique().max()

    # Initialize empty column
    match_data['distance_to_ball'] = None

    # Set starting index to 0
    start_iloc = 0
    for match in range(1, number_of_matches+1):

        # Store match length (in frames/rows), set ending index
        match_len = len(ball_data[ball_data['match'] == match])
        end_iloc = start_iloc+match_len*4

        # Get ball coordinates
        ballx = ball_data[ball_data['match'] == match].pos_x.reset_index(drop=True)
        bally = ball_data[ball_data['match'] == match].pos_y.reset_index(drop=True)
        ballz = ball_data[ball_data['match'] == match].pos_z.reset_index(drop=True)

        # Get player coordinates
        matchx = match_data[match_data['match'] == match].pos_x.reset_index(drop=True)
        matchy = match_data[match_data['match'] == match].pos_y.reset_index(drop=True)
        matchz = match_data[match_data['match'] == match].pos_z.reset_index(drop=True)

        # Calculate distance using distance()
        match_data['distance_to_ball'].iloc[start_iloc:end_iloc] = distance(match_len, ballx, bally, ballz, matchx, matchy, matchz)
        
        # Reset starting index
        start_iloc = end_iloc
        

    return match_data


'''
Adding `closest_player`, `closest_player_distance`, and `most_recent_hitter` columns to ball_data

WORK FOR FUTURE 3v3 and 1v1 GAME DATA:
* Generalize getting the player's name and distance for 1v1 and 2v2 games *
* i.e. avoid directly accessing: p1 = current_match.player.unique()[0] *
'''

def closest_player_hitter(ball_data, match_data):

    # Get the number of matches
    number_of_matches = match_data.match.unique().max()
    for match in range(1, number_of_matches+1):
        
        # Isolate match
        current_match = match_data[match_data['match'] == match]

        # Get each player's name and distance column
        p1 = current_match.player.unique()[0]
        p1_distance = current_match[current_match['player'] == p1].distance_to_ball.values

        p2 = current_match.player.unique()[1]
        p2_distance = current_match[current_match['player'] == p2].distance_to_ball.values

        p3 = current_match.player.unique()[2]
        p3_distance = current_match[current_match['player'] == p3].distance_to_ball.values

        p4 = current_match.player.unique()[3]
        p4_distance = current_match[current_match['player'] == p4].distance_to_ball.values

        # Stack their distances together for easy comparison
        stack = np.stack((p1_distance, p2_distance, p3_distance, p4_distance), axis=1)

        # Calculate closest_player_distance
        closest_player_distance_temp = np.min(stack, axis=1)

        # Find closest_player
        conditions = [p1_distance <= closest_player_distance_temp, 
                    p2_distance <= closest_player_distance_temp, 
                    p3_distance <= closest_player_distance_temp, 
                    p4_distance <= closest_player_distance_temp]

        choices = [p1, p2, p3, p4]

        closest_player_temp = np.select(conditions, choices, default='None')

        # Store numpy arrays, concatinate after match 1
        if match == 1:
            closest_player_distance = closest_player_distance_temp
            closest_player = closest_player_temp
        else:
            closest_player_distance = np.concatenate((closest_player_distance, closest_player_distance_temp))
            closest_player = np.concatenate((closest_player, closest_player_temp))

    

    # Create 'closest_player', 'closest_player_distance' columns
    ball_data['closest_player'] = closest_player
    ball_data['closest_player_distance'] = closest_player_distance

    # Set conditions for a registered hit
    hit_condition = (ball_data['closest_player_distance'] < 250) & ball_data['vel_x'].notna()

    # Create 'most_recent_hitter' column
    ball_data['most_recent_hitter'] = np.where(hit_condition, ball_data['closest_player'], 'None')

    return ball_data
