# %%
import pandas as pd

sheet_id = "1_dBxF4MrsxyKmTm1hez2316YO-UzCqccIKJZiwSnXgg"

sheets = {
    "Teams": "0",
    "Availability": "407075336",
    "Groups": "731501686",
    "Team Match Results": "2123782057",
    "Player Match Results": "543999939"
}


def load(sheet_name):
    if sheet_name in sheets:
        gid = sheets[sheet_name]
        url = f"https://docs.google.com/spreadsheets/d/{
            sheet_id}/export?format=csv&gid={gid}"
        return pd.read_csv(url)
    else:
        raise Exception("Sheet name not known")


firearm_types = ['Assault', 'Pistol',
                 'RocketLauncher', 'SMG', 'Shotgun', 'Sniper']
firearm_names = {
    'Assault': ['AKM', 'CX4', 'FAL', 'M60', 'MK18', 'RFB'],
    'Pistol': ['1911', '357', 'Px4', 'Tec9'],
    'RocketLauncher': ['SMAW'],
    'SMG': ['MP5', 'MP9', 'P90', 'UMP', 'Uzi'],
    'Shotgun': ['DT11', 'M1014', 'Matador'],
    'Sniper': ['AWP', 'Sako85']
}


def load_all(filtered=False):
    team_df = load("Teams")
    availability_df = load("Availability")
    groups_df = load("Groups")
    team_match_results_df = load("Team Match Results")
    player_match_results_df = load("Player Match Results")

    if filtered:
        # List of weapon names to exclude
        exclude_weapons = ['MG338', 'M590', 'MK12']

        # Create a regex pattern to match columns related to these weapons
        exclude_pattern = '|'.join(
            [f'Firearm.*{weapon}.*' for weapon in exclude_weapons])

        # Filter out the columns that match the exclude pattern
        filtered_player_match_results_df = player_match_results_df.filter(
            regex=f'^(?!{exclude_pattern}).*')

        player_match_results_df = filtered_player_match_results_df.copy()

        del filtered_player_match_results_df

    return team_df, availability_df, groups_df, team_match_results_df, player_match_results_df, firearm_types, firearm_names


# %%
%load_ext intheloop
# %%
team_df, availability_df, groups_df, team_match_results_df, player_match_results_df, firearm_types, firearm_names = load_all()

# %%
% % ai

Make me a pretty chart showing who's the best of the best with the rocket launcher


# %%
