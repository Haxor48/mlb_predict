from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from . import predict
##import predict
from sklearn import preprocessing
from pybaseball import batting_stats, pitching_stats, fielding_stats, team_batting, team_pitching, standings, schedule_and_record, bwar_bat, bwar_pitch, \
    statcast_pitcher_exitvelo_barrels, statcast_pitcher_expected_stats, statcast_pitcher_pitch_arsenal, statcast_batter_exitvelo_barrels, \
    statcast_batter_expected_stats, statcast_outs_above_average, statcast_sprint_speed, statcast_outfielder_jump, cache

min_max_scalar = preprocessing.MinMaxScaler()
IndexBatter = namedtuple('IndexBatter', ['Name', 'Team', 'G', 'AVG', 'OBP', 'SLG', 'OPS', 'wOBA', 'wRC', 'fWAR', 'bWAR'])
IndexPitcher = namedtuple('IndexBatter', ['Name', 'Team', 'G', 'IP', 'ERA', 'WHIP', 'K9', 'KBB', 'FIP', 'SIERA', 'fWAR', 'bWAR'])
Team = namedtuple('Team', ['Name', 'W', 'L', 'WPer', 'GB', 'Streak', 'RunsF', 'RunsA', 'Diff', 'Pyth'])
StandardBatter = namedtuple('StandardBatter', ['Name', 'Team', 'Pos', 'G', 'PA', 'H', 'HR', 'R', 'RBI', 'SB', 'BB', 'AVG', 'OBP', 'SLG', 'OPS', 'wRC', 'wOBA', 'OFF', 'fWAR', 'bWAR', 'aWAR', 'WPA'])
StandardPitcher = namedtuple('StandardPitcher', ['Name', 'Team', 'G', 'GS', 'IP', 'K', 'BB', 'K9', 'BB9', 'HR9', 'KBB', 'AVG', 'RS9', 'WHIP', 'ERA', 'FIP', 'xFIP', 'SIERA', 'fWAR', 'bWAR', 'aWAR', 'WPA'])
TeamBatter = namedtuple('TeamBatter', ['Team', 'Year', 'H', 'HR', 'R', 'RBI', 'SB', 'BB', 'AVG', 'OBP', 'SLG', 'OPS', 'wRC', 'wOBA'])
TeamPitcher = namedtuple('TeamPitcher', ['Team', 'Year', 'K', 'BB', 'K9', 'BB9', 'HR9', 'KBB', 'AVG', 'RS9', 'WHIP', 'ERA', 'FIP', 'xFIP', 'SIERA'])
StandardFielder = namedtuple('StandardFielder', ['Name', 'Team', 'Pos', 'G', 'GS', 'IP', 'PO', 'A', 'DP', 'E', 'ARM', 'UZR', 'DRS', 'OAA', 'FRM', 'DEF', 'R', 'SB', 'CS', 'BsR'])
PlayerStandardFielder = namedtuple('PlayerStandardFielder', ['Name', 'Team', 'Year', 'Pos', 'G', 'GS', 'IP', 'PO', 'A', 'DP', 'E', 'ARM', 'UZR', 'DRS', 'OAA', 'FRM', 'DEF', 'R', 'SB', 'CS', 'BsR'])
BasicBatter = namedtuple('BasicBatter', ['Name', 'Team', 'Pos', 'G', 'AB', 'PA', 'H', 'Doubles', 'Triples', 'HR', 'BB', 'HBP', 'TB', 'R', 'RBI', 'SB', 'CS', 'SO', 'GDP', 'AVG', 'OBP', 'SLG', 'OPS'])
AdvancedBatter = namedtuple('AdvancedBatter', ['Name', 'Team', 'Pos', 'G', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'wRC', 'wOBA', 'ISO', 'BABIP', 'Clutch', 'BB', 'K', 'xBA', 'OFF', 'DEF', 'fWAR', 'bWAR', 'aWAR', 'WPA'])
BasicPitcher = namedtuple('BasicPitcher', ['Name', 'Team', 'G', 'GS', 'IP', 'W', 'L', 'CG', 'SV', 'BS', 'ER', 'H', 'BB', 'SO', 'ERA', 'WHIP'])
AdvancedPitcher = namedtuple('AdvancedPitcher', ['Name', 'Team', 'G', 'IP', 'K9', 'BB9', 'HR9', 'KBB', 'AVG', 'Clutch', 'BABIP', 'LOB', 'WHIP', 'ERA', 'RS9', 'FIP', 'xFIP', 'SIERA', 'fWAR', 'bWAR', 'aWAR', 'WPA'])
StatcastHitter = namedtuple('StatcastHitter', ['Name', 'Team' , 'Pos', 'PA', 'Barrel', 'HardHit', 'LD', 'GB', 'FB', 'O_Swing', 'Z_Swing', 'Contact', 'Pull', 'Cent', 'Oppo', 'Whiff', 'Avg_EV', 'Max_EV', 'LA', 'xBA', 'xSLG', 'xwOBA', 'BABIP'])
StatcastFielder = namedtuple('StatcastFielder', ['Name', 'Team', 'Pos', 'OAA', 'OAAIn', 'OAA3B', 'OAA1B', 'OAABack', 'OAASuccess', 'OAAEstSuccess', 'FldRuns', 'Reaction', 'Burst', 'Route', 'Ft', 'Sprint', 'HP1B', 'Bolts'])
PlayerBasicBatter = namedtuple('PlayerBasicBatter', ['Name', 'Team', 'Year', 'Pos', 'G', 'AB', 'PA', 'H', 'Doubles', 'Triples', 'HR', 'BB', 'HBP', 'TB', 'R', 'RBI', 'SB', 'CS', 'SO', 'GDP', 'AVG', 'OBP', 'SLG', 'OPS'])
PlayerAdvancedBatter = namedtuple('PlayerAdvancedBatter', ['Name', 'Team', 'Year', 'Pos', 'G', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'wRC', 'wOBA', 'ISO', 'BABIP', 'Clutch', 'BB', 'K', 'xBA', 'OFF', 'DEF', 'fWAR', 'bWAR', 'aWAR', 'WPA'])
PlayerBasicPitcher = namedtuple('PlayerBasicPitcher', ['Name', 'Team', 'Year', 'G', 'GS', 'IP', 'W', 'L', 'CG', 'SV', 'BS', 'ER', 'H', 'BB', 'SO', 'ERA', 'WHIP'])
PlayerAdvancedPitcher = namedtuple('PlayerAdvancedPitcher', ['Name', 'Team', 'Year', 'G', 'IP', 'K9', 'BB9', 'HR9', 'KBB', 'AVG', 'Clutch', 'BABIP', 'LOB', 'WHIP', 'ERA', 'RS9', 'FIP', 'xFIP', 'SIERA', 'fWAR', 'bWAR', 'aWAR', 'WPA'])
PlayerStatcastHitter = namedtuple('PlayerStatcastHitter', ['Name', 'Team' , 'Year', 'Pos', 'PA', 'Barrel', 'HardHit', 'LD', 'GB', 'FB', 'O_Swing', 'Z_Swing', 'Contact', 'Pull', 'Cent', 'Oppo', 'Whiff', 'Avg_EV', 'Max_EV', 'LA', 'xBA', 'xSLG', 'xwOBA', 'BABIP'])
PlayerStatcastFielder = namedtuple('PlayerStatcastFielder', ['Name', 'Team', 'Year', 'Pos', 'OAA', 'OAAIn', 'OAA3B', 'OAA1B', 'OAABack', 'OAASuccess', 'OAAEstSuccess', 'FldRuns', 'Reaction', 'Burst', 'Route', 'Ft', 'Sprint', 'HP1B', 'Bolts'])
PlayerStandardBatter = namedtuple('PlayerStandardBatter', ['Name', 'Team', 'Year', 'Pos', 'G', 'PA', 'H', 'HR', 'R', 'RBI', 'SB', 'BB', 'AVG', 'OBP', 'SLG', 'OPS', 'wRC', 'wOBA', 'OFF', 'fWAR', 'bWAR', 'aWAR', 'WPA'])
PlayerStandardPitcher = namedtuple('PlayerStandardPitcher', ['Name', 'Team', 'Year', 'G', 'GS', 'IP', 'K', 'BB', 'K9', 'BB9', 'HR9', 'KBB', 'AVG', 'RS9', 'WHIP', 'ERA', 'FIP', 'xFIP', 'SIERA', 'fWAR', 'bWAR', 'aWAR', 'WPA'])
TeamFull = namedtuple('TeamFull', ['Team', 'W', 'L', 'WPercent', 'GB', 'Strk', 'Diff', 'Rf', 'RfG', 'Ra', 'RaG', 'PythW', 'PythL', 'HomeWL', 'RoadWL', 'Over500WL', 'Under500WL', 'ProjRf', 'ProjRa', 'ProjDiff', 'ProjW', 'ProjL'])
StatcastPitcher = namedtuple('StatcastPitcher', ['Name', 'Team', 'G', 'IP', 'HardHit', 'Barrel', 'K', 'BB', 'LD', 'GB', 'FB', 'KBB', 'GBFB', 'HRFB', 'Soft', 'Med', 'AvrEV', 'MaxEV', 'NumPitches', 'MaxVelo', 'AvgSpin', 'xBA', 'xSLG', 'xwOBA', 'xERA'])
PlayerStatcastPitcher = namedtuple('PlayerStatcastPitcher', ['Name', 'Team', 'Year', 'G', 'IP', 'HardHit', 'Barrel', 'K', 'BB', 'LD', 'GB', 'FB', 'KBB', 'GBFB', 'HRFB', 'Soft', 'Med', 'AvrEV', 'MaxEV', 'NumPitches', 'MaxVelo', 'AvgSpin', 'xBA', 'xSLG', 'xwOBA', 'xERA'])
PlayerInfo = namedtuple('PlayerInfo', ['Name', 'Team', 'Pos', 'Age', 'Img'])
TeamInfo = namedtuple('TeamInfo', ['Team', 'Img', 'WL', 'LastYrWL', 'PythWL', 'ProjWL'])
PlayerSalary = namedtuple('PlayerSalary', ['Name', 'Age', 'War', 'Salary', 'ProjSalary'])
TopPercentile = namedtuple('TopPercentile', ['Name', 'Percent', 'Val'])

batting_fields = ['G', 'PA', 'H', 'HR', 'R', 'RBI', 'SB', 'BB', 'AVG', 'OBP', 'SLG', 'OPS', 'wRC+', 'wOBA', 'Bat', 'WAR', 'bWAR', 'aWAR', 'WPA']
pitching_fields = ['G', 'GS', 'IP', 'SO', 'BB', 'K/9', 'BB/9', 'HR/9', 'K/BB', 'AVG', 'WHIP', 'RS/9', 'ERA', 'FIP', 'xFIP', 'SIERA', 'WAR', 'bWAR', 'aWAR', 'WPA']

team_abbr = {'New York Yankees': 'NYY', 'Toronto Blue Jays': 'TOR', 'Baltimore Orioles': 'BAL', 'Tampa Bay Rays': 'TBR', 'Boston Red Sox': 'BOS', 'Cleveland Guardians': 'CLE', 'Chicago White Sox': 'CHW', 'Minnesota Twins': 'MIN', \
    'Detroit Tigers': 'DET', 'Kansas City Royals': 'KCR', 'Houston Astros': 'HOU', 'Seattle Mariners': 'SEA', 'Los Angeles Angels': 'LAA', 'Texas Rangers': 'TEX', 'Oakland Athletics': 'OAK', \
    'Atlanta Braves': 'ATL', 'New York Mets': 'NYM', 'Philadelphia Phillies': 'PHI', 'Miami Marlins': 'MIA', 'Washington Nationals': 'WSN', 'St. Louis Cardinals': 'STL', 'Milwaukee Brewers': 'MIL', \
    'Chicago Cubs': 'CHC', 'Cincinnati Reds': 'CIN', 'Pittsburgh Pirates': 'PIT', 'Los Angeles Dodgers': 'LAD', 'San Diego Padres': 'SDP', 'San Francisco Giants': 'SFG', 'Arizona Diamondbacks': 'ARI', \
    'Colorado Rockies': 'COL'}

hitting_stat_ref = ['H', 'HR', 'BB', 'SO', 'SB', 'AVG', 'BB%', 'K%', 'BB/K', 'OBP', 'ISO', 'BABIP', 'GB/FB', 'LD%', 'GB%', 'FB%', 'wOBA', 'wRC+', 'O-Swing%', 'Swing%', 'Contact%', 'SwStr%', 'BsR', 'Def', 'Pull%', \
    'Oppo%', 'Soft%', 'EV', 'LA', 'maxEV', 'Barrel%', 'HardHit%', 'xBA', 'aWAR', 'Salary']
pitching_stat_ref = ['W', 'ERA', 'IP', 'BB', 'SO', 'K/9', 'BB/9', 'K/BB', 'HR/9', 'AVG', 'WHIP', 'BABIP', 'LOB%', 'LD%', 'GB%', 'FB%', 'FIP', 'xFIP', 'Zone%', 'Contact%', 'SwStr%', 'O-Swing%', 'K%', 'BB%', 'SIERA', \
    'Soft%', 'EV', 'Barrel%', 'maxEV', 'HardHit%', 'xERA', 'aWAR', 'AvrSpin', 'MaxVelo', 'Salary', 'NumPitches', 'est_ba', 'est_slg', 'est_woba']
fielding_stat_ref = ['PO', 'A', 'E', 'ARM', 'DRS', 'OAA', 'FRM']
fielding_stat_statcast_ref = ['fielding_runs_prevented', 'sprint_speed', 'hp_to_1b']

class Data:
    def __init__(self):
        self._player_batting = {}
        self._player_pitching = {}
        self._player_fielding = {}
        self._predicted_percentiles_batter = {}
        self._predicted_percentiles_pitcher = {}
        self._predicted_batter = {}
        self._predicted_pitcher = {}
        self._batting_bwar = None
        self._pitching_bwar = None
        self._team_hitting = {}
        self._team_pitching = {}
        self._standing = {}
        self._statcast_pitcher = {}
        self._statcast_batter = {}
        self._statcast_fielder = {}
        self._team_schedule = {}
        self._base_url = ''
        self._proj_salaries = None
        self._batting_percentiles = {}
        self._pitching_percentiles = {}
        self._fielding_percentiles = {}
        self._fielding_statcast_percentiles = {}
        self._team_img_urls = {}

    def get_data(self, url, hitting: bool, pitching: bool, fielding: bool, team_hitting: bool, team_pitching: bool, standings: bool, schedule: bool, team_imgs: bool):
        cache.enable()
        path = Path("./ml_b/data")
        if not any(path.iterdir()):
            self._load_from_apis(hitting, pitching, fielding, team_hitting, team_pitching, standings, schedule, team_imgs)
        else:
            self._load_from_files(hitting, pitching, fielding, team_hitting, team_pitching, standings, schedule, team_imgs)
        self._base_url = url
        
    def save_dataframes(self, hitting: bool, pitching: bool, fielding: bool, team_hitting: bool, team_pitching: bool, standings: bool, schedule: bool, team_imgs: bool):
        if hitting:
            path = Path("./ml_b/data/batting_bwar.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._batting_bwar.to_csv(path)
        if pitching:
            path = Path("./ml_b/data/pitching_bwar.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._pitching_bwar.to_csv(path)
        for i in range(2015, 2023):
            if standings:
                for j in range(len(self._standing[i])):
                    path = Path("./ml_b/data/standings_" + str(i) + "_" + str(j) + ".csv")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self._standing[i][j].to_csv(path)
            if team_hitting:
                path = Path("./ml_b/data/team_hitting_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._team_hitting[i].to_csv(path)
            if team_pitching:
                path = Path("./ml_b/data/team_pitching_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._team_pitching[i].to_csv(path)
            if hitting:
                path = Path("./ml_b/data/player_batting_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._player_batting[i].to_csv(path)
                path = Path("./ml_b/data/statcast_batter_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._statcast_batter[i].to_csv(path)
                path = Path("./ml_b/data/percentile_hitter_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._batting_percentiles[i].to_csv(path)
                path = Path("./ml_b/data/predicted_percentile_hitter_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._predicted_percentiles_batter[i].to_csv(path)
                path = Path("./ml_b/data/predicted_hitter_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._predicted_batter[i].to_csv(path)
            if pitching:
                path = Path("./ml_b/data/player_pitching_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._player_pitching[i].to_csv(path)
                path = Path("./ml_b/data/statcast_pitcher_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._statcast_pitcher[i].to_csv(path)
                path = Path("./ml_b/data/percentile_pitcher_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._pitching_percentiles[i].to_csv(path)
                path = Path("./ml_b/data/predicted_percentile_pitcher_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._predicted_percentiles_pitcher[i].to_csv(path)
                path = Path("./ml_b/data/predicted_pitcher_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._predicted_pitcher[i].to_csv(path)
            if fielding:
                path = Path("./ml_b/data/player_fielding_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._player_fielding[i].to_csv(path)
                path = Path("./ml_b/data/statcast_fielder_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._statcast_fielder[i].to_csv(path)
                path = Path("./ml_b/data/percentile_fielder_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._fielding_percentiles[i].to_csv(path)
                path = Path("./ml_b/data/percentile_fielder_statcast_" + str(i) + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._fielding_statcast_percentiles[i].to_csv(path)
        if schedule:
            for key in team_abbr.keys():
                path = Path("./ml_b/data/team_schedule_" + team_abbr[key] + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                self._team_schedule[team_abbr[key]].to_csv(path)
        if hitting and pitching:
            path = Path("./ml_b/data/proj_salaries.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._proj_salaries.to_csv(path)
        if team_imgs:
            self._save_team_img_urls()
        
    def _load_from_files(self, hitting: bool, pitching: bool, fielding: bool, team_hitting: bool, team_pitching: bool, standings: bool, schedule: bool, team_imgs: bool):
        if hitting:
            self._batting_bwar = pd.read_csv("./ml_b/data/batting_bwar.csv")
        if pitching:
            self._pitching_bwar = pd.read_csv("./ml_b/data/pitching_bwar.csv")
        for i in range(2015, 2023):
            if standings:
                self._standing[i] = []
                for j in range(6):
                    self._standing[i].append(pd.read_csv("./ml_b/data/standings_" + str(i) + "_" + str(j) + ".csv"))
            if team_hitting:
                self._team_hitting[i] = pd.read_csv("./ml_b/data/team_hitting_" + str(i) + ".csv")
            if team_pitching:
                self._team_pitching[i] = pd.read_csv("./ml_b/data/team_pitching_" + str(i) + ".csv")
            if hitting:
                self._player_batting[i] = pd.read_csv("./ml_b/data/player_batting_" + str(i) + ".csv")
                self._statcast_batter[i] = pd.read_csv("./ml_b/data/statcast_batter_" + str(i) + ".csv")
                self._batting_percentiles[i] = pd.read_csv("./ml_b/data/percentile_hitter_" + str(i) + ".csv")
                self._predicted_percentiles_batter[i] = pd.read_csv("./ml_b/data/predicted_percentile_hitter_" + str(i) + ".csv")
                self._predicted_batter[i] = pd.read_csv("./ml_b/data/predicted_hitter_" + str(i) + ".csv")
            if pitching:
                self._player_pitching[i] = pd.read_csv("./ml_b/data/player_pitching_" + str(i) + ".csv")
                self._statcast_pitcher[i] = pd.read_csv("./ml_b/data/statcast_pitcher_" + str(i) + ".csv")
                self._pitching_percentiles[i] = pd.read_csv("./ml_b/data/percentile_pitcher_" + str(i) + ".csv")
                self._predicted_percentiles_pitcher[i] = pd.read_csv("./ml_b/data/predicted_percentile_pitcher_" + str(i) + ".csv")
                self._predicted_pitcher[i] = pd.read_csv("./ml_b/data/predicted_pitcher_" + str(i) + ".csv")
            if fielding:
                self._player_fielding[i] = pd.read_csv("./ml_b/data/player_fielding_" + str(i) + ".csv")
                self._statcast_fielder[i] = pd.read_csv("./ml_b/data/statcast_fielder_" + str(i) + ".csv")
                self._fielding_percentiles[i] = pd.read_csv("./ml_b/data/percentile_fielder_" + str(i) + ".csv")
                self._fielding_statcast_percentiles[i] = pd.read_csv("./ml_b/data/percentile_fielder_statcast_" + str(i) + ".csv")
        if schedule:
            for key in team_abbr.keys():
                self._team_schedule[team_abbr[key]] = pd.read_csv("./ml_b/data/team_schedule_" + team_abbr[key] + ".csv")
        if team_imgs:
            self._load_team_img_urls(True)
        if pitching and hitting:
            self._proj_salaries = pd.read_csv('./ml_b/data/proj_salaries.csv')
        
    def _load_from_apis(self, hitting: bool, pitching: bool, fielding: bool, team_hitting: bool, team_pitch: bool, standing: bool, schedule: bool, team_imgs: bool):
        if hitting:
            self._batting_bwar = bwar_bat()
            self._batting_bwar = self._batting_bwar[self._batting_bwar['year_ID'] > 2014]
        if pitching:
            self._pitching_bwar = bwar_pitch()
            self._pitching_bwar = self._pitching_bwar[self._pitching_bwar['year_ID'] > 2014]
        for i in range(2015, 2023):
            if pitching:
                self._statcast_pitcher[i] = pd.merge(pd.merge(pd.merge(statcast_pitcher_exitvelo_barrels(i, 1), statcast_pitcher_expected_stats(i, 1), on=['first_name', 'last_name'], how='outer'), \
                    statcast_pitcher_pitch_arsenal(i, minP=1, arsenal_type="avg_spin"), on=['first_name', 'last_name'], how='outer'), statcast_pitcher_pitch_arsenal(i, minP=1), on=['first_name', 'last_name'], how='outer')
                self._give_fg_names_pitch(i)
                self._player_pitching[i] = pd.merge(pitching_stats(i, qual=1), self._statcast_pitcher[i], on='Name', how='outer').drop_duplicates()
                self._set_extra_pitcher_stats(i)
            if hitting:
                self._statcast_batter[i] = pd.merge(statcast_batter_exitvelo_barrels(i, 1), statcast_batter_expected_stats(i, 1), on=['first_name', 'last_name'], how='outer')
                self._player_batting[i] = batting_stats(i, qual=1).drop_duplicates()
                self._set_extra_batter_stats(i)
            if fielding:
                self._statcast_fielder[i] = pd.merge(pd.merge(statcast_outs_above_average(i, 'all', 1), statcast_sprint_speed(i, 1), on=['first_name', 'last_name'], how='outer'), \
                    statcast_outfielder_jump(i, 1), on=['first_name', 'last_name'], how='outer')
                self._player_fielding[i] = fielding_stats(i, qual=10)
                self._set_extra_fielder_stats(i)
            if team_hitting:
                self._team_hitting[i] = team_batting(i)
            if team_pitch:
                self._team_pitching[i] = team_pitching(i)
            if standing:
                self._standing[i] = standings(i)
            if hitting and pitching:
                self._save_percentiles(i)
                self._get_predicted_percentiles(i)
                self._predicted_percentiles_to_vals(i)
        if schedule:
            self._set_team_schedule(self._standing[2022])
        if team_imgs:
            self._load_team_img_urls(False)
        if hitting and pitching:
            self._remove_all_accents()
            self._set_salaries()
            self.predict_sals()
        self.save_dataframes(hitting, pitching, fielding, team_hitting, team_pitch, standing, schedule, team_imgs)
        
    def _save_percentiles(self, year: int):
        cur_batting = self._player_batting[year]
        cur_pitching = self._player_pitching[year]
        cur_fielding = self._player_fielding[year]
        cur_fielding_statcast = self._statcast_fielder[year]
        cur_batting = cur_batting.drop(['Dol', 'Age Rng', 'FLDPos', 'player_id'], axis=1)
        age_index = cur_batting.columns.get_loc('Age')
        batting_vals = cur_batting.values[:, age_index:]
        batter_percents = min_max_scalar.fit_transform(batting_vals)
        self._batting_percentiles[year] = pd.concat([cur_batting.iloc[:, :age_index], pd.DataFrame(batter_percents, columns=cur_batting.columns[age_index:]), self._player_batting[year][['FLDPos']]], axis=1)
        cur_pitching = cur_pitching.drop(['Dollars', 'Age Rng', 'player_id', 'last_name', 'first_name'], axis=1)
        age_index = cur_pitching.columns.get_loc('Age')
        pitching_vals = cur_pitching.values[:, age_index:]
        pitcher_percents = min_max_scalar.fit_transform(pitching_vals)
        self._pitching_percentiles[year] = pd.concat([cur_pitching.iloc[:, :age_index],pd.DataFrame( pitcher_percents, columns=cur_pitching.columns[age_index:])], axis=1)
        age_index = cur_fielding.columns.get_loc('G')
        fielding_vals = cur_fielding.values[:, age_index:]
        fielder_percents = min_max_scalar.fit_transform(fielding_vals)
        self._fielding_percentiles[year] = pd.concat([cur_fielding.iloc[:, :age_index], pd.DataFrame(fielder_percents, columns=cur_fielding.columns[age_index:])], axis=1)
        cur_fielding_statcast = cur_fielding_statcast.drop(cur_fielding_statcast.columns[[x for x in range(cur_fielding_statcast.columns.get_loc('actual_success_rate_formatted'), cur_fielding_statcast.columns.get_loc('age'))]], axis=1)
        age_index = cur_fielding_statcast.columns.get_loc('fielding_runs_prevented')
        fielding_statcast_vals = cur_fielding_statcast.values[:, age_index:]
        fielder_statcast_percents = min_max_scalar.fit_transform(fielding_statcast_vals)
        self._fielding_statcast_percentiles[year] = pd.concat([cur_fielding_statcast.iloc[:, :age_index], pd.DataFrame(fielder_statcast_percents, columns=cur_fielding_statcast.columns[age_index:])], axis=1)
        
    def _set_team_schedule(self, standings: list):
        for i in range(len(standings)):
            for r in range(len(standings[i])):
                abbr = team_abbr[standings[i].iloc[r]['Tm']]
                self._team_schedule[abbr] = schedule_and_record(2022, abbr)
                
    def _save_team_img_urls(self):
        path = Path('./ml_b/data/team_img_urls.txt')
        writer = path.open('w')
        for key in self._team_img_urls.keys():
            writer.write(key + ' ' + self._team_img_urls[key] + '\n')
        writer.close()
        
    def _predicted_percentiles_to_vals(self, year):
        batters = self._player_batting[year]
        batters = batters[batting_fields]
        self._predicted_batter[year] = self._player_batting[year][['Name', 'Team', 'FLDPos']]
        vals = batters.values
        vals = np.nan_to_num(vals)
        vals = vals.astype(float)
        batter_vals = vals.view()
        min_max_scalar.fit(batter_vals)
        percentiles = self._predicted_percentiles_batter[year].values[:,3:]
        percentiles = np.nan_to_num(percentiles)
        percentile_vals = percentiles.view()
        un_percentile = min_max_scalar.inverse_transform(percentile_vals)
        self._predicted_batter[year] = pd.concat([self._predicted_batter[year], pd.DataFrame(un_percentile, columns=batting_fields)], axis=1)
        pitchers = self._player_pitching[year]
        pitchers = pitchers[pitching_fields]
        self._predicted_pitcher[year] = self._player_pitching[year][['Name', 'Team']]
        vals = pitchers.values
        vals = np.nan_to_num(vals)
        vals = vals.astype(float)
        pitcher_vals = vals.view()
        min_max_scalar.fit(pitcher_vals)
        percentiles = self._predicted_percentiles_pitcher[year].values[:,2:]
        percentiles = np.nan_to_num(percentiles)
        percentile_vals = percentiles.view()
        un_percentile = min_max_scalar.inverse_transform(percentile_vals)
        self._predicted_pitcher[year] = pd.concat([self._predicted_pitcher[year], pd.DataFrame(un_percentile, columns=pitching_fields)], axis=1)
        
    def _get_predicted_percentiles(self, year):
        self._predicted_percentiles_batter[year] = self._player_batting[year][['Name', 'Team', 'FLDPos']]
        self._predicted_percentiles_pitcher[year] = self._player_pitching[year][['Name', 'Team']]
        for field in batting_fields:
            predicted = predict.predict_batter(self, field, year)
            self._predicted_percentiles_batter[year] = pd.concat([self._predicted_percentiles_batter[year], pd.DataFrame(predicted, columns=[field])], axis=1)
        for field in pitching_fields:
            predicted = predict.predict_pitcher(self, field, year)
            self._predicted_percentiles_pitcher[year]= pd.concat([self._predicted_percentiles_pitcher[year], pd.DataFrame(predicted, columns=[field])], axis=1)
                
    def _load_team_img_urls(self, from_file: bool):
        if from_file:
            path = Path('./ml_b/data/team_img_urls.txt')
            reader = path.open('r')
            for line in reader.readlines():
                if len(line) > 0:
                    split = line.split(' ')
                    self._team_img_urls[split[0]] = split[1]
            reader.close()
        else:
            for value in team_abbr.values():
                res = requests.get(f'https://www.baseball-reference.com/teams/{value}/2022.shtml')
                soup = BeautifulSoup(res.text, 'html.parser')
                logo = soup.find_all(class_ = 'teamlogo')[0]['src']
                self._team_img_urls[value] = logo
                
    def _get_player_img_url(self, name: str) -> str:
        player_id = ''
        if self.is_hitter(name):
            player_id = self._player_batting[2022][self._player_batting[2022]['Name'] == name].iloc[0]['player_id']
        else:
            player_id = self._player_pitching[2022][self._player_pitching[2022]['Name'] == name].iloc[0]['player_id']
        res = requests.get(f'https://www.baseball-reference.com/players/{player_id[0]}/{player_id}.shtml')
        soup  = BeautifulSoup(res.text, 'html.parser')
        player = soup.find_all('img')[1]['src']
        return player
    
    def get_players(self) -> list[PlayerInfo]:
        batting = self._player_batting[2022].sort_values(by='Name', ascending=True)
        pitching = self._player_pitching[2022].sort_values(by='Name', ascending=True)
        output = []
        bat_count = 0
        pitch_count = 0
        while bat_count < len(batting) or pitch_count < len(pitching):
            if bat_count >= len(batting):
                output.append(PlayerInfo(pitching.iloc[pitch_count]['Name'], pitching.iloc[pitch_count]['Team'], 'P', pitching.iloc[pitch_count]['Age'], ''))
                pitch_count += 1
            elif pitch_count >= len(pitching):
                output.append(PlayerInfo(batting.iloc[bat_count]['Name'], batting.iloc[bat_count]['Team'], batting.iloc[bat_count]['FLDPos'], batting.iloc[bat_count]['Age'], ''))
                bat_count += 1
            elif batting.iloc[bat_count]['Name'] > pitching.iloc[pitch_count]['Name']:
                output.append(PlayerInfo(pitching.iloc[pitch_count]['Name'], pitching.iloc[pitch_count]['Team'], 'P', pitching.iloc[pitch_count]['Age'], ''))
                pitch_count += 1
            else:
                output.append(PlayerInfo(batting.iloc[bat_count]['Name'], batting.iloc[bat_count]['Team'], batting.iloc[bat_count]['FLDPos'], batting.iloc[bat_count]['Age'], ''))
                bat_count += 1
        return output
    
    def predict_sals(self):
        players = pd.concat([self._player_batting[2022][['Name', 'Age', 'aWAR', 'Salary']], self._player_pitching[2022][['Name', 'Age', 'aWAR', 'Salary']]])
        self._proj_salaries = players.join(pd.DataFrame(predict.predict_salaries(self), columns=['ProjSalary']), how='outer')
        
    def get_url(self):
        return self._base_url
    
    def _percent(self, val: float) -> int:
        try:
            return int(round(val, 2) * 100)
        except:
            return 0
    
    def get_player_batting(self, year: int):
        return self._player_batting[year]
    
    def get_player_pitching(self, year: int):
        return self._player_pitching[year]
    
    def get_player_top_percentiles(self, name: str, is_hitter: bool) -> list[list[TopPercentile]]:
        output = [[], []]
        if is_hitter:
            hitters = self._batting_percentiles[2022]
            hitters_real = self._player_batting[2022]
            hitter = hitters[hitters['Name'] == name]
            for stat in hitting_stat_ref:
                percent = self._percent(hitter.iloc[0][stat])
                if len(output[0]) < 5:
                    if len(output[0]) == 0:
                        output[0].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                    else:
                        if percent > output[0][0].Percent:
                            output[0].insert(0, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        elif percent < output[0][len(output[0])-1].Percent:
                            output[0].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        else:
                            for i in range(len(output[0])-1, 0, -1):
                                if percent <= output[0][i-1].Percent:
                                    output[0].insert(i, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                                    break
                else:
                    if percent > output[0][4].Percent:
                        output[0].pop()
                        if percent > output[0][0].Percent:
                            output[0].insert(0, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        elif percent < output[0][len(output[0])-1].Percent:
                            output[0].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        else:
                            for i in range(len(output[0])-1, 0, -1):
                                if percent <= output[0][i-1].Percent:
                                    output[0].insert(i, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                                    break
                if len(output[1]) < 5:
                    if len(output[1]) == 0:
                        output[1].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                    else:
                        if percent < output[1][0].Percent:
                            output[1].insert(0, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        elif percent > output[1][len(output[1])-1].Percent:
                            output[1].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        else:
                            for i in range(len(output[1])-1, 0, -1):
                                if percent >= output[1][i-1].Percent:
                                    output[1].insert(i, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                                    break
                else:
                    if percent < output[1][4].Percent:
                        output[1].pop()
                        if percent < output[1][0].Percent:
                            output[1].insert(0, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        elif percent > output[1][len(output[1])-1].Percent:
                            output[1].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        else:
                            for i in range(len(output[1])-1, 0, -1):
                                if percent >= output[1][i-1].Percent:
                                    output[1].insert(i, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                                    break
            hitters = self._fielding_percentiles[2022]
            hitters_real = self._player_fielding[2022]
            hitter = hitters[hitters['Name'] == name]
            for stat in fielding_stat_ref:
                percent = self._percent(hitter.iloc[0][stat])
                if len(output[0]) < 5:
                    if len(output[0]) == 0:
                        output[0].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                    else:
                        if percent > output[0][0].Percent:
                            output[0].insert(0, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        elif percent < output[0][len(output[0])-1].Percent:
                            output[0].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        else:
                            for i in range(len(output[0])-1, 0, -1):
                                if percent <= output[0][i-1].Percent:
                                    output[0].insert(i, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                                    break
                else:
                    if percent > output[0][4].Percent:
                        output[0].pop()
                        if percent > output[0][0].Percent:
                            output[0].insert(0, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        elif percent < output[0][len(output[0])-1].Percent:
                            output[0].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        else:
                            for i in range(len(output[0])-1, 0, -1):
                                if percent <= output[0][i-1].Percent:
                                    output[0].insert(i, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                                    break
                if len(output[1]) < 5:
                    if len(output[1]) == 0:
                        output[1].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                    else:
                        if percent < output[1][0].Percent:
                            output[1].insert(0, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        elif percent > output[1][len(output[1])-1].Percent:
                            output[1].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        else:
                            for i in range(len(output[1])-1, 0, -1):
                                if percent >= output[1][i-1].Percent:
                                    output[1].insert(i, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                                    break
                else:
                    if percent < output[1][4].Percent:
                        output[1].pop()
                        if percent < output[1][0].Percent:
                            output[1].insert(0, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        elif percent > output[1][len(output[1])-1].Percent:
                            output[1].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                        else:
                            for i in range(len(output[1])-1, 0, -1):
                                if percent >= output[1][i-1].Percent:
                                    output[1].insert(i, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitters_real[hitters_real['Name'] == name].iloc[0][stat]))
                                    break
            hitters = self._fielding_statcast_percentiles[2022]
            hitters_real = self._statcast_fielder[2022]
            split = name.split(' ')
            hitter = hitters[(hitters['first_name'] == split[0]) & (hitters['last_name'] == split[1])]
            hitter_real = hitters_real[(hitters_real['first_name'] == split[0]) & (hitters_real['last_name'] == split[1])]
            for stat in fielding_stat_statcast_ref:
                percent = self._percent(hitter.iloc[0][stat])
                if len(output[0]) < 5:
                    if len(output[0]) == 0:
                        output[0].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitter_real.iloc[0][stat]))
                    else:
                        if percent > output[0][0].Percent:
                            output[0].insert(0, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitter_real.iloc[0][stat]))
                        elif percent < output[0][len(output[0])-1].Percent:
                            output[0].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitter_real.iloc[0][stat]))
                        else:
                            for i in range(len(output[0])-1, 0, -1):
                                if percent <= output[0][i-1].Percent:
                                    output[0].insert(i, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitter_real.iloc[0][stat]))
                                    break
                else:
                    if percent > output[0][4].Percent:
                        output[0].pop()
                        if percent > output[0][0].Percent:
                            output[0].insert(0, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitter_real.iloc[0][stat]))
                        elif percent < output[0][len(output[0])-1].Percent:
                            output[0].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitter_real.iloc[0][stat]))
                        else:
                            for i in range(len(output[0])-1, 0, -1):
                                if percent <= output[0][i-1].Percent:
                                    output[0].insert(i, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitter_real.iloc[0][stat]))
                                    break
                if len(output[1]) < 5:
                    if len(output[1]) == 0:
                        output[1].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitter_real.iloc[0][stat]))
                    else:
                        if percent < output[1][0].Percent:
                            output[1].insert(0, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitter_real.iloc[0][stat]))
                        elif percent > output[1][len(output[1])-1].Percent:
                            output[1].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitter_real.iloc[0][stat]))
                        else:
                            for i in range(len(output[1])-1, 0, -1):
                                if percent >= output[1][i-1].Percent:
                                    output[1].insert(i, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitter_real.iloc[0][stat]))
                                    break
                else:
                    if percent < output[1][4].Percent:
                        output[1].pop()
                        if percent < output[1][0].Percent:
                            output[1].insert(0, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitter_real.iloc[0][stat]))
                        elif percent > output[1][len(output[1])-1].Percent:
                            output[1].append(TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitter_real.iloc[0][stat]))
                        else:
                            for i in range(len(output[1])-1, 0, -1):
                                if percent >= output[1][i-1].Percent:
                                    output[1].insert(i, TopPercentile(stat, self._percent(hitter.iloc[0][stat]), hitter_real.iloc[0][stat]))
                                    break
        else:
            pitchers = self._pitching_percentiles[2022]
            pitchers_real = self._player_pitching[2022]
            pitcher = pitchers[pitchers['Name'] == name]
            for stat in pitching_stat_ref:
                percent = self._percent(pitcher.iloc[0][stat])
                if len(output[0]) < 5:
                    if len(output[0]) == 0:
                        output[0].append(TopPercentile(stat, self._percent(pitcher.iloc[0][stat]), pitchers_real[pitchers_real['Name'] == name].iloc[0][stat]))
                    else:
                        if percent > output[0][0].Percent:
                            output[0].insert(0, TopPercentile(stat, self._percent(pitcher.iloc[0][stat]), pitchers_real[pitchers_real['Name'] == name].iloc[0][stat]))
                        elif percent < output[0][len(output[0])-1].Percent:
                            output[0].append(TopPercentile(stat, self._percent(pitcher.iloc[0][stat]), pitchers_real[pitchers_real['Name'] == name].iloc[0][stat]))
                        else:
                            for i in range(len(output[0])-1, 0, -1):
                                if percent <= output[0][i-1].Percent:
                                    output[0].insert(i, TopPercentile(stat, self._percent(pitcher.iloc[0][stat]), pitchers_real[pitchers_real['Name'] == name].iloc[0][stat]))
                                    break
                else:
                    if percent > output[0][4].Percent:
                        output[0].pop()
                        if percent > output[0][0].Percent:
                            output[0].insert(0, TopPercentile(stat, self._percent(pitcher.iloc[0][stat]), pitchers_real[pitchers_real['Name'] == name].iloc[0][stat]))
                        elif percent < output[0][len(output[0])-1].Percent:
                            output[0].append(TopPercentile(stat, self._percent(pitcher.iloc[0][stat]), pitchers_real[pitchers_real['Name'] == name].iloc[0][stat]))
                        else:
                            for i in range(len(output[0])-1, 0, -1):
                                if percent <= output[0][i-1].Percent:
                                    output[0].insert(i, TopPercentile(stat, self._percent(pitcher.iloc[0][stat]), pitchers_real[pitchers_real['Name'] == name].iloc[0][stat]))
                                    break
                if len(output[1]) < 5:
                    if len(output[1]) == 0:
                        output[1].append(TopPercentile(stat, self._percent(pitcher.iloc[0][stat]), pitchers_real[pitchers_real['Name'] == name].iloc[0][stat]))
                    else:
                        if percent < output[1][0].Percent:
                            output[1].insert(0, TopPercentile(stat, self._percent(pitcher.iloc[0][stat]), pitchers_real[pitchers_real['Name'] == name].iloc[0][stat]))
                        elif percent > output[1][len(output[1])-1].Percent:
                            output[1].append(TopPercentile(stat, self._percent(pitcher.iloc[0][stat]), pitchers_real[pitchers_real['Name'] == name].iloc[0][stat]))
                        else:
                            for i in range(len(output[1])-1, 0, -1):
                                if percent >= output[1][i-1].Percent:
                                    output[1].insert(i, TopPercentile(stat, self._percent(pitcher.iloc[0][stat]), pitchers_real[pitchers_real['Name'] == name].iloc[0][stat]))
                                    break
                else:
                    if percent < output[1][4].Percent:
                        output[1].pop()
                        if percent < output[1][0].Percent:
                            output[1].insert(0, TopPercentile(stat, self._percent(pitcher.iloc[0][stat]), pitchers_real[pitchers_real['Name'] == name].iloc[0][stat]))
                        elif percent > output[1][len(output[1])-1].Percent:
                            output[1].append(TopPercentile(stat, self._percent(pitcher.iloc[0][stat]), pitchers_real[pitchers_real['Name'] == name].iloc[0][stat]))
                        else:
                            for i in range(len(output[1])-1, 0, -1):
                                if percent >= output[1][i-1].Percent:
                                    output[1].insert(i, TopPercentile(stat, self._percent(pitcher.iloc[0][stat]), pitchers_real[pitchers_real['Name'] == name].iloc[0][stat]))
                                    break
        return output
    
    def get_player_salaries(self) -> list[PlayerSalary]:
        output = []
        salaries = self._proj_salaries.dropna().sort_values(by='ProjSalary', ascending=False)
        for i in range(len(salaries)):
            output.append(PlayerSalary(salaries.iloc[i]['Name'], salaries.iloc[i]['Age'], salaries.iloc[i]['aWAR'], salaries.iloc[i]['Salary'], salaries.iloc[i]['ProjSalary']))
        return output
    
    def get_player_info(self, name: str) -> PlayerInfo:
        if self.is_hitter(name):
            player_data = self._player_batting[2022][self._player_batting[2022]['Name'] == name].iloc[0]
            return PlayerInfo(name, player_data['Team'], player_data['FLDPos'], player_data['Age'], self._get_player_img_url(name))
        player_data = self._player_pitching[2022][self._player_pitching[2022]['Name'] == name].iloc[0]
        return PlayerInfo(name, player_data['Team'], 'P', player_data['Age'], self._get_player_img_url(name))
    
    def get_team_imgs(self):
        return self._team_img_urls
    
    def get_team_info(self, team: str) -> TeamInfo:
        full_name = list(team_abbr.keys())[list(team_abbr.values()).index(team)]
        team_cur = pd.DataFrame()
        team_past = pd.DataFrame()
        for i in range(len(self._standing[2022])):
            to_break = False
            for j in range(len(self._standing[2022][i])):
                if self._standing[2022][i].iloc[j]['Tm'] == full_name:
                    team_cur = self._standing[2022][i].iloc[j]
                    to_break = True
                    break
            if to_break:
                break
        for i in range(len(self._standing[2021])):
            to_break = False
            for j in range(len(self._standing[2021][i])):
                if self._standing[2021][i].iloc[j]['Tm'] == full_name:
                    team_past = self._standing[2021][i].iloc[j]
                    to_break = True
                    break
            if to_break:
                break
        rf = self._team_hitting[2022][self._team_hitting[2022]['Team'] == team].iloc[0]['R']
        ra = self._team_pitching[2022][self._team_pitching[2022]['Team'] == team].iloc[0]['R']
        pyth_w = self._calc_pyth_wins(rf, ra)
        return TeamInfo(full_name, self._team_img_urls[team], str(team_cur['W']) + ' - '  + str(team_cur['L']), str(team_past['W']) + ' - ' + str(team_past['L']), f'{pyth_w}-{162-pyth_w}', str(team_cur['W']) + ' - ' + str(team_cur['L']))
        
    def get_home_hitting_leaders(self) -> list[IndexBatter]:
        output = []
        current_hitters = self._player_batting[2022].sort_values(by='WAR', ascending=False).iloc[0:10]
        for i in range(len(current_hitters)):
            output.append(IndexBatter(current_hitters.iloc[i]['Name'], current_hitters.iloc[i]['Team'], current_hitters.iloc[i]['G'], current_hitters.iloc[i]['AVG'], \
                                    current_hitters.iloc[i]['OBP'], current_hitters.iloc[i]['SLG'], current_hitters.iloc[i]['OPS'], current_hitters.iloc[i]['wOBA'], \
                                    current_hitters.iloc[i]['wRC+'], current_hitters.iloc[i]['WAR'], current_hitters.iloc[i]['bWAR']))
        return output

    def get_home_pitching_leaders(self) -> list[IndexBatter]:
        output = []
        current_pitchers = self._player_pitching[2022].sort_values(by='WAR', ascending=False).iloc[0:10]
        for i in range(len(current_pitchers)):
            output.append(IndexPitcher(current_pitchers.iloc[i]['Name'], current_pitchers.iloc[i]['Team'], current_pitchers.iloc[i]['G'], current_pitchers.iloc[i]['IP'], \
                current_pitchers.iloc[i]['ERA'], current_pitchers.iloc[i]['WHIP'], current_pitchers.iloc[i]['K/9'], current_pitchers.iloc[i]['K/BB'], \
                current_pitchers.iloc[i]['FIP'], current_pitchers.iloc[i]['SIERA'], current_pitchers.iloc[i]['WAR'], current_pitchers.iloc[i]['bWAR']))
        return output

    def get_standings(self, division: int) -> list[Team]:
        output = []
        for i in range(len(self._standing[2022][division])):
            abbr = team_abbr[self._standing[2022][division].iloc[i]['Tm']]
            rf = self._team_hitting[2022][self._team_hitting[2022]['Team'] == abbr].iloc[0]['R']
            ra = self._team_pitching[2022][self._team_pitching[2022]['Team'] == abbr].iloc[0]['R']
            pyth_w = self._calc_pyth_wins(rf, ra)
            output.append(Team(self._standing[2022][division].iloc[i]['Tm'], self._standing[2022][division].iloc[i]['W'], self._standing[2022][division].iloc[i]['L'], self._standing[2022][division].iloc[i]['W-L%'], self._standing[2022][division].iloc[i]['GB'], \
                self._get_streak(self._standing[2022][division].iloc[i]['Tm']), rf, ra, (rf-ra), str(pyth_w) + '-' + str(162-pyth_w)))
        return output
    
    def get_projected_hitting_leaders(self, sortBy: str, start: int, end: int, ascend: bool, percentiles: bool) -> list[StandardBatter]:
        if start < 0:
            start = 0
        if start >= len(self._predicted_batter[2022]):
            start = len(self._predicted_batter[2022]) - 1
        if end < 0:
            end = 0
        if end >= len(self._predicted_batter[2022]):
            end = len(self._predicted_batter[2022]) - 1
        if start > end:
            end = start
        output = []
        current_hitters = None
        if percentiles:
            current_hitters = self._predicted_percentiles_batter[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        else:
            current_hitters = self._predicted_batter[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        for i in range(len(current_hitters)):
            if percentiles:
                output.append(StandardBatter(current_hitters.iloc[i]['Name'], current_hitters.iloc[i]['Team'], current_hitters.iloc[i]['FLDPos'], \
                        self._percent(current_hitters.iloc[i]['G']), self._percent(current_hitters.iloc[i]['PA']), self._percent(current_hitters.iloc[i]['H']), self._percent(current_hitters.iloc[i]['HR']), self._percent(current_hitters.iloc[i]['R']), self._percent(current_hitters.iloc[i]['RBI']), \
                        self._percent(current_hitters.iloc[i]['SB']), self._percent(current_hitters.iloc[i]['BB']), self._percent(current_hitters.iloc[i]['AVG']), self._percent(current_hitters.iloc[i]['OBP']), self._percent(current_hitters.iloc[i]['SLG']), self._percent(current_hitters.iloc[i]['OPS']), \
                        self._percent(current_hitters.iloc[i]['wRC+']), self._percent(current_hitters.iloc[i]['wOBA']), self._percent(current_hitters.iloc[i]['Bat']), self._percent(current_hitters.iloc[i]['WAR']), self._percent(current_hitters.iloc[i]['bWAR']), self._percent(current_hitters.iloc[i]['aWAR']), \
                        self._percent(current_hitters.iloc[i]['WPA'])))
            else:
                output.append(StandardBatter(current_hitters.iloc[i]['Name'], current_hitters.iloc[i]['Team'], current_hitters.iloc[i]['FLDPos'], \
                        int(current_hitters.iloc[i]['G']), int(current_hitters.iloc[i]['PA']), int(current_hitters.iloc[i]['H']), int(current_hitters.iloc[i]['HR']), int(current_hitters.iloc[i]['R']), int(current_hitters.iloc[i]['RBI']), \
                        int(current_hitters.iloc[i]['SB']), int(current_hitters.iloc[i]['BB']), round(current_hitters.iloc[i]['AVG'], 3), round(current_hitters.iloc[i]['OBP'], 3), round(current_hitters.iloc[i]['SLG'], 3), round(current_hitters.iloc[i]['OPS'], 3), \
                        int(current_hitters.iloc[i]['wRC+']), round(current_hitters.iloc[i]['wOBA'], 3), round(current_hitters.iloc[i]['Bat'], 1), round(current_hitters.iloc[i]['WAR'], 1), round(current_hitters.iloc[i]['bWAR'], 1), round(current_hitters.iloc[i]['aWAR'], 1), \
                        round(current_hitters.iloc[i]['WPA'], 2)))
        return output
    
    def get_player_projected_hitting(self, name: str, percentiles: bool) -> list[PlayerStandardBatter]:
        output = []
        for i in range(2022, 2014, -1):
            if name not in list(self._predicted_batter[i]['Name']):
                continue
            if percentiles:
                hitter = self._predicted_percentiles_batter[i][self._predicted_percentiles_batter[i]['Name'] == name]
                output.append(PlayerStandardBatter(hitter.iloc[0]['Name'], hitter.iloc[0]['Team'], i, hitter.iloc[0]['FLDPos'], \
                        self._percent(hitter.iloc[0]['G']), self._percent(hitter.iloc[0]['PA']), self._percent(hitter.iloc[0]['H']), self._percent(hitter.iloc[0]['HR']), self._percent(hitter.iloc[0]['R']), self._percent(hitter.iloc[0]['RBI']), \
                        self._percent(hitter.iloc[0]['SB']), self._percent(hitter.iloc[0]['BB']), self._percent(hitter.iloc[0]['AVG']), self._percent(hitter.iloc[0]['OBP']), self._percent(hitter.iloc[0]['SLG']), self._percent(hitter.iloc[0]['OPS']), \
                        self._percent(hitter.iloc[0]['wRC+']), self._percent(hitter.iloc[0]['wOBA']), self._percent(hitter.iloc[0]['Bat']), self._percent(hitter.iloc[0]['WAR']), self._percent(hitter.iloc[0]['bWAR']), self._percent(hitter.iloc[0]['aWAR']), \
                        self._percent(hitter.iloc[0]['WPA'])))
            else:
                hitter = self._predicted_batter[i][self._predicted_batter[i]['Name'] == name]
                output.append(PlayerStandardBatter(hitter.iloc[0]['Name'], hitter.iloc[0]['Team'], i, hitter.iloc[0]['FLDPos'], \
                        int(hitter.iloc[0]['G']), int(hitter.iloc[0]['PA']), int(hitter.iloc[0]['H']), int(hitter.iloc[0]['HR']), int(hitter.iloc[0]['R']), int(hitter.iloc[0]['RBI']), \
                        int(hitter.iloc[0]['SB']), int(hitter.iloc[0]['BB']), round(hitter.iloc[0]['AVG'], 3), round(hitter.iloc[0]['OBP'], 3), round(hitter.iloc[0]['SLG'], 3), round(hitter.iloc[0]['OPS'], 3), \
                        int(hitter.iloc[0]['wRC+']), round(hitter.iloc[0]['wOBA'], 3), round(hitter.iloc[0]['Bat'], 1), round(hitter.iloc[0]['WAR'], 1), round(hitter.iloc[0]['bWAR'], 1), round(hitter.iloc[0]['aWAR'], 1), \
                        round(hitter.iloc[0]['WPA'], 2)))
        return output
    
    def get_projected_pitching_leaders(self, sortBy: str, start: int, end: int, ascend: bool, percentiles: bool) -> list[StandardPitcher]:
        if start < 0:
            start = 0
        if start >= len(self._predicted_pitcher[2022]):
            start = len(self._predicted_pitcher[2022]) - 1
        if end < 0:
            end = 0
        if end >= len(self._predicted_pitcher[2022]):
            end = len(self._predicted_pitcher[2022]) - 1
        if start > end:
            end = start
        output = []
        current_pitchers = None
        if percentiles:
            current_pitchers = self._predicted_percentiles_pitcher[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        else:
            current_pitchers = self._predicted_pitcher[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        for i in range(len(current_pitchers)):
            if percentiles:
                output.append(StandardPitcher(current_pitchers.iloc[i]['Name'], current_pitchers.iloc[i]['Team'], self._percent(current_pitchers.iloc[i]['G']), self._percent(current_pitchers.iloc[i]['GS']), self._percent(current_pitchers.iloc[i]['IP']), \
                        self._percent(current_pitchers.iloc[i]['SO']), self._percent(current_pitchers.iloc[i]['BB']), self._percent(current_pitchers.iloc[i]['K/9']), self._percent(current_pitchers.iloc[i]['BB/9']), self._percent(current_pitchers.iloc[i]['HR/9']), self._percent(current_pitchers.iloc[i]['K/BB']), \
                        self._percent(current_pitchers.iloc[i]['AVG']), self._percent(current_pitchers.iloc[i]['RS/9']), self._percent(current_pitchers.iloc[i]['WHIP']), self._percent(current_pitchers.iloc[i]['ERA']), self._percent(current_pitchers.iloc[i]['FIP']), self._percent(current_pitchers.iloc[i]['xFIP']), \
                        self._percent(current_pitchers.iloc[i]['SIERA']), self._percent(current_pitchers.iloc[i]['WAR']), self._percent(current_pitchers.iloc[i]['bWAR']), self._percent(current_pitchers.iloc[i]['aWAR']), self._percent(current_pitchers.iloc[i]['WPA'])))
            else:
                output.append(StandardPitcher(current_pitchers.iloc[i]['Name'], current_pitchers.iloc[i]['Team'], int(current_pitchers.iloc[i]['G']), int(current_pitchers.iloc[i]['GS']), int(current_pitchers.iloc[i]['IP']), \
                        int(current_pitchers.iloc[i]['SO']), int(current_pitchers.iloc[i]['BB']), round(current_pitchers.iloc[i]['K/9'], 1), round(current_pitchers.iloc[i]['BB/9'], 1), round(current_pitchers.iloc[i]['HR/9'], 1), round(current_pitchers.iloc[i]['K/BB'], 1), \
                        round(current_pitchers.iloc[i]['AVG'], 3), round(current_pitchers.iloc[i]['RS/9'], 2), round(current_pitchers.iloc[i]['WHIP'], 2), round(current_pitchers.iloc[i]['ERA'], 2), round(current_pitchers.iloc[i]['FIP'], 2), round(current_pitchers.iloc[i]['xFIP'], 2), \
                        round(current_pitchers.iloc[i]['SIERA'], 2), round(current_pitchers.iloc[i]['WAR'], 1), round(current_pitchers.iloc[i]['bWAR'], 1), round(current_pitchers.iloc[i]['aWAR'], 1), round(current_pitchers.iloc[i]['WPA'], 2)))
        return output
    
    def get_player_projected_pitching(self, name: str, percentiles: bool) -> list[PlayerStandardPitcher]:
        output = []
        for i in range(2022, 2014, -1):
            if name not in list(self._predicted_pitcher[i]['Name']):
                continue
            if percentiles:
                pitcher = self._predicted_percentiles_pitcher[i][self._predicted_percentiles_pitcher[i]['Name'] == name]
                output.append(PlayerStandardPitcher(pitcher.iloc[0]['Name'], pitcher.iloc[0]['Team'], i, self._percent(pitcher.iloc[0]['G']), self._percent(pitcher.iloc[0]['GS']), self._percent(pitcher.iloc[0]['IP']), \
                        self._percent(pitcher.iloc[0]['SO']), self._percent(pitcher.iloc[0]['BB']), self._percent(pitcher.iloc[0]['K/9']), self._percent(pitcher.iloc[0]['BB/9']), self._percent(pitcher.iloc[0]['HR/9']), self._percent(pitcher.iloc[0]['K/BB']), \
                        self._percent(pitcher.iloc[0]['AVG']), self._percent(pitcher.iloc[0]['RS/9']), self._percent(pitcher.iloc[0]['WHIP']), self._percent(pitcher.iloc[0]['ERA']), self._percent(pitcher.iloc[0]['FIP']), self._percent(pitcher.iloc[0]['xFIP']), \
                        self._percent(pitcher.iloc[0]['SIERA']), self._percent(pitcher.iloc[0]['WAR']), self._percent(pitcher.iloc[0]['bWAR']), self._percent(pitcher.iloc[0]['aWAR']), self._percent(pitcher.iloc[0]['WPA'])))
            else:
                pitcher = self._predicted_pitcher[i][self._predicted_pitcher[i]['Name'] == name]
                output.append(PlayerStandardPitcher(pitcher.iloc[0]['Name'], pitcher.iloc[0]['Team'], i, int(pitcher.iloc[0]['G']), int(pitcher.iloc[0]['GS']), int(pitcher.iloc[0]['IP']), \
                        int(pitcher.iloc[0]['SO']), int(pitcher.iloc[0]['BB']), round(pitcher.iloc[0]['K/9'], 1), round(pitcher.iloc[0]['BB/9'], 1), round(pitcher.iloc[0]['HR/9'], 1), round(pitcher.iloc[0]['K/BB'], 1), \
                        round(pitcher.iloc[0]['AVG'], 3), round(pitcher.iloc[0]['RS/9'], 2), round(pitcher.iloc[0]['WHIP'], 2), round(pitcher.iloc[0]['ERA'], 2), round(pitcher.iloc[0]['FIP'], 2), round(pitcher.iloc[0]['xFIP'], 2), \
                        round(pitcher.iloc[0]['SIERA'], 2), round(pitcher.iloc[0]['WAR'], 1), round(pitcher.iloc[0]['bWAR'], 1), round(pitcher.iloc[0]['aWAR'], 1), round(pitcher.iloc[0]['WPA'], 2)))
        return output
    
    def get_standard_hitting_leaders(self, sortBy: str, start: int, end: int, ascend: bool) -> list[StandardBatter]:
        if start < 0:
            start = 0
        if start >= len(self._player_batting[2022]):
            start = len(self._player_batting[2022]) - 1
        if end < 0:
            end = 0
        if end >= len(self._player_batting[2022]):
            end = len(self._player_batting[2022]) - 1
        if start > end:
            end = start
        output = []
        current_hitters = self._player_batting[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        for i in range(len(current_hitters)):
            output.append(StandardBatter(current_hitters.iloc[i]['Name'], current_hitters.iloc[i]['Team'], current_hitters.iloc[i]['FLDPos'], \
                        current_hitters.iloc[i]['G'], current_hitters.iloc[i]['PA'], current_hitters.iloc[i]['H'], current_hitters.iloc[i]['HR'], current_hitters.iloc[i]['R'], current_hitters.iloc[i]['RBI'], \
                        current_hitters.iloc[i]['SB'], current_hitters.iloc[i]['BB'], current_hitters.iloc[i]['AVG'], current_hitters.iloc[i]['OBP'], current_hitters.iloc[i]['SLG'], current_hitters.iloc[i]['OPS'], \
                        current_hitters.iloc[i]['wRC+'], current_hitters.iloc[i]['wOBA'], current_hitters.iloc[i]['Bat'], current_hitters.iloc[i]['WAR'], current_hitters.iloc[i]['bWAR'], current_hitters.iloc[i]['aWAR'], \
                        current_hitters.iloc[i]['WPA']))
        return output
    
    def get_standard_pitching_leaders(self, sortBy: str, start: int, end: int, ascend: bool) -> list[StandardPitcher]:
        if start < 0:
            start = 0
        if start >= len(self._player_pitching[2022]):
            start = len(self._player_pitching[2022]) - 1
        if end < 0:
            end = 0
        if end >= len(self._player_pitching[2022]):
            end = len(self._player_pitching[2022]) - 1
        if start > end:
            end = start
        output = []
        current_pitchers = self._player_pitching[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        for i in range(len(current_pitchers)):
            output.append(StandardPitcher(current_pitchers.iloc[i]['Name'], current_pitchers.iloc[i]['Team'], current_pitchers.iloc[i]['G'], current_pitchers.iloc[i]['GS'], current_pitchers.iloc[i]['IP'], \
                        current_pitchers.iloc[i]['SO'], current_pitchers.iloc[i]['BB'], current_pitchers.iloc[i]['K/9'], current_pitchers.iloc[i]['BB/9'], current_pitchers.iloc[i]['HR/9'], current_pitchers.iloc[i]['K/BB'], \
                        current_pitchers.iloc[i]['AVG'], current_pitchers.iloc[i]['RS/9'], current_pitchers.iloc[i]['WHIP'], current_pitchers.iloc[i]['ERA'], current_pitchers.iloc[i]['FIP'], current_pitchers.iloc[i]['xFIP'], \
                        current_pitchers.iloc[i]['SIERA'], current_pitchers.iloc[i]['WAR'], current_pitchers.iloc[i]['bWAR'], current_pitchers.iloc[i]['aWAR'], current_pitchers.iloc[i]['WPA']))
        return output
    
    def get_team_hitting(self, team: str) -> list[TeamBatter]:
        output = []
        for i in range(2022, 2014, -1):
            current_hitters = self._team_hitting[i]
            current_hitters = current_hitters[current_hitters['Team'] == team]
            output.append(TeamBatter(current_hitters.iloc[0]['Team'], i, current_hitters.iloc[0]['H'], current_hitters.iloc[0]['HR'], current_hitters.iloc[0]['R'], current_hitters.iloc[0]['RBI'], \
                        current_hitters.iloc[0]['SB'], current_hitters.iloc[0]['BB'], current_hitters.iloc[0]['AVG'], current_hitters.iloc[0]['OBP'], current_hitters.iloc[0]['SLG'], current_hitters.iloc[0]['OPS'], \
                        current_hitters.iloc[0]['wRC+'], current_hitters.iloc[0]['wOBA']))
        return output
    
    def get_team_pitching(self, team: str) -> list[TeamPitcher]:
        output = []
        for i in range(2022, 2014, -1):
            current_pitchers = self._team_pitching[i]
            current_pitchers = current_pitchers[current_pitchers['Team'] == team]
            output.append(TeamPitcher(current_pitchers.iloc[0]['Team'], i, current_pitchers.iloc[0]['SO'], current_pitchers.iloc[0]['BB'], current_pitchers.iloc[0]['K/9'], current_pitchers.iloc[0]['BB/9'], current_pitchers.iloc[0]['HR/9'], current_pitchers.iloc[0]['K/BB'], \
                        current_pitchers.iloc[0]['AVG'], current_pitchers.iloc[0]['RS/9'], current_pitchers.iloc[0]['WHIP'], current_pitchers.iloc[0]['ERA'], current_pitchers.iloc[0]['FIP'], current_pitchers.iloc[0]['xFIP'], \
                        current_pitchers.iloc[0]['SIERA']))
        return output
    
    def get_team_hitting_leaders(self, team: str) -> list[StandardBatter]:
        output = []
        current_hitters = self._player_batting[2022].sort_values(by='PA', ascending=False)
        current_hitters = current_hitters[current_hitters['Team'] == team]
        for i in range(len(current_hitters)):
            output.append(StandardBatter(current_hitters.iloc[i]['Name'], current_hitters.iloc[i]['Team'], current_hitters.iloc[i]['FLDPos'], \
                        current_hitters.iloc[i]['G'], current_hitters.iloc[i]['PA'], current_hitters.iloc[i]['H'], current_hitters.iloc[i]['HR'], current_hitters.iloc[i]['R'], current_hitters.iloc[i]['RBI'], \
                        current_hitters.iloc[i]['SB'], current_hitters.iloc[i]['BB'], current_hitters.iloc[i]['AVG'], current_hitters.iloc[i]['OBP'], current_hitters.iloc[i]['SLG'], current_hitters.iloc[i]['OPS'], \
                        current_hitters.iloc[i]['wRC+'], current_hitters.iloc[i]['wOBA'], current_hitters.iloc[i]['Bat'], current_hitters.iloc[i]['WAR'], current_hitters.iloc[i]['bWAR'], current_hitters.iloc[i]['aWAR'], \
                        current_hitters.iloc[i]['WPA']))
        return output
    
    def get_team_pitching_leaders(self, team: str) -> list[StandardPitcher]:
        output = []
        current_pitchers = self._player_pitching[2022].sort_values(by='IP', ascending=False)
        current_pitchers = current_pitchers[current_pitchers['Team'] == team]
        for i in range(len(current_pitchers)):
            output.append(StandardPitcher(current_pitchers.iloc[i]['Name'], current_pitchers.iloc[i]['Team'], current_pitchers.iloc[i]['G'], current_pitchers.iloc[i]['GS'], current_pitchers.iloc[i]['IP'], \
                        current_pitchers.iloc[i]['SO'], current_pitchers.iloc[i]['BB'], current_pitchers.iloc[i]['K/9'], current_pitchers.iloc[i]['BB/9'], current_pitchers.iloc[i]['HR/9'], current_pitchers.iloc[i]['K/BB'], \
                        current_pitchers.iloc[i]['AVG'], current_pitchers.iloc[i]['RS/9'], current_pitchers.iloc[i]['WHIP'], current_pitchers.iloc[i]['ERA'], current_pitchers.iloc[i]['FIP'], current_pitchers.iloc[i]['xFIP'], \
                        current_pitchers.iloc[i]['SIERA'], current_pitchers.iloc[i]['WAR'], current_pitchers.iloc[i]['bWAR'], current_pitchers.iloc[i]['aWAR'], current_pitchers.iloc[i]['WPA']))
        return output
    
    def get_team_fielding_leaders(self, team: str) -> list[StandardFielder]:
        output = []
        current_fielders = self._player_fielding[2022].sort_values(by='Inn', ascending=False)
        current_fielders = current_fielders[current_fielders['Team'] == team]
        for i in range(len(current_fielders)):
            output.append(StandardFielder(current_fielders.iloc[i]['Name'], current_fielders.iloc[i]['Team'], current_fielders.iloc[i]['Pos'], current_fielders.iloc[i]['G'], current_fielders.iloc[i]['GS'], \
                    current_fielders.iloc[i]['Inn'], current_fielders.iloc[i]['PO'], current_fielders.iloc[i]['A'], current_fielders.iloc[i]['DP'], current_fielders.iloc[i]['E'], current_fielders.iloc[i]['ARM'], \
                    current_fielders.iloc[i]['UZR'], current_fielders.iloc[i]['DRS'], current_fielders.iloc[i]['OAA'], current_fielders.iloc[i]['FRM'], current_fielders.iloc[i]['Def'], current_fielders.iloc[i]['R'], \
                    current_fielders.iloc[i]['BRSB'], current_fielders.iloc[i]['BRCS'], current_fielders.iloc[i]['BsR']))
        return output
    
    def get_standard_fielding_leaders(self, sortBy: str, start: int, end: int, ascend: bool, percentiles: bool) -> list[StandardFielder]:
        if start < 0:
            start = 0
        if start >= len(self._player_fielding[2022]):
            start = len(self._player_fielding[2022]) - 1
        if end < 0:
            end = 0
        if end >= len(self._player_fielding[2022]):
            end = len(self._player_fielding[2022]) - 1
        if start > end:
            end = start
        output = []
        current_fielders = None
        if percentiles:
            current_fielders = self._fielding_percentiles[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        else:
            current_fielders = self._player_fielding[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        for i in range(len(current_fielders)):
            if percentiles:
                output.append(StandardFielder(current_fielders.iloc[i]['Name'], current_fielders.iloc[i]['Team'], current_fielders.iloc[i]['Pos'], self._percent(current_fielders.iloc[i]['G']), self._percent(current_fielders.iloc[i]['GS']), \
                        self._percent(current_fielders.iloc[i]['Inn']), self._percent(current_fielders.iloc[i]['PO']), self._percent(current_fielders.iloc[i]['A']), self._percent(current_fielders.iloc[i]['DP']), self._percent(current_fielders.iloc[i]['E']), self._percent(current_fielders.iloc[i]['ARM']), \
                        self._percent(current_fielders.iloc[i]['UZR']), self._percent(current_fielders.iloc[i]['DRS']), self._percent(current_fielders.iloc[i]['OAA']), self._percent(current_fielders.iloc[i]['FRM']), self._percent(current_fielders.iloc[i]['Def']), self._percent(current_fielders.iloc[i]['R']), \
                        self._percent(current_fielders.iloc[i]['BRSB']), self._percent(current_fielders.iloc[i]['BRCS']), self._percent(current_fielders.iloc[i]['BsR'])))
            else:
                output.append(StandardFielder(current_fielders.iloc[i]['Name'], current_fielders.iloc[i]['Team'], current_fielders.iloc[i]['Pos'], current_fielders.iloc[i]['G'], current_fielders.iloc[i]['GS'], \
                        current_fielders.iloc[i]['Inn'], current_fielders.iloc[i]['PO'], current_fielders.iloc[i]['A'], current_fielders.iloc[i]['DP'], current_fielders.iloc[i]['E'], current_fielders.iloc[i]['ARM'], \
                        current_fielders.iloc[i]['UZR'], current_fielders.iloc[i]['DRS'], current_fielders.iloc[i]['OAA'], current_fielders.iloc[i]['FRM'], current_fielders.iloc[i]['Def'], current_fielders.iloc[i]['R'], \
                        current_fielders.iloc[i]['BRSB'], current_fielders.iloc[i]['BRCS'], current_fielders.iloc[i]['BsR']))
        return output
    
    def get_player_fielding_standard(self, name: str, percentiles: bool):
        output = []
        for i in range(2022, 2014, -1):
            if name not in list(self._player_batting[i]['Name']):
                continue
            if percentiles:
                fielder = self._fielding_percentiles[i][self._fielding_percentiles[i]['Name'] == name]
                output.append(PlayerStandardFielder(fielder.iloc[0]['Name'], fielder.iloc[0]['Team'], i, fielder.iloc[0]['Pos'], self._percent(fielder.iloc[0]['G']), self._percent(fielder.iloc[0]['GS']), \
                        self._percent(fielder.iloc[0]['Inn']), self._percent(fielder.iloc[0]['PO']), self._percent(fielder.iloc[0]['A']), self._percent(fielder.iloc[0]['DP']), self._percent(fielder.iloc[0]['E']), self._percent(fielder.iloc[0]['ARM']), \
                        self._percent(fielder.iloc[0]['UZR']), self._percent(fielder.iloc[0]['DRS']), self._percent(fielder.iloc[0]['OAA']), self._percent(fielder.iloc[0]['FRM']), self._percent(fielder.iloc[0]['Def']), self._percent(fielder.iloc[0]['R']), \
                        self._percent(fielder.iloc[0]['BRSB']), self._percent(fielder.iloc[0]['BRCS']), self._percent(fielder.iloc[0]['BsR'])))
            else:
                fielder = self._player_fielding[i][self._player_fielding[i]['Name'] == name]
                output.append(PlayerStandardFielder(fielder.iloc[0]['Name'], fielder.iloc[0]['Team'], i, fielder.iloc[0]['Pos'], fielder.iloc[0]['G'], fielder.iloc[0]['GS'], \
                        fielder.iloc[0]['Inn'], fielder.iloc[0]['PO'], fielder.iloc[0]['A'], fielder.iloc[0]['DP'], fielder.iloc[0]['E'], fielder.iloc[0]['ARM'], \
                        fielder.iloc[0]['UZR'], fielder.iloc[0]['DRS'], fielder.iloc[0]['OAA'], fielder.iloc[0]['FRM'], fielder.iloc[0]['Def'], fielder.iloc[0]['R'], \
                        fielder.iloc[0]['BRSB'], fielder.iloc[0]['BRCS'], fielder.iloc[0]['BsR']))
        return output
    
    def get_basic_batter_leaders(self, sortBy: str, start: int, end: int, ascend: bool, percentiles: bool) -> list[BasicBatter]:
        if start < 0:
            start = 0
        if start >= len(self._player_batting[2022]):
            start = len(self._player_batting[2022]) - 1
        if end < 0:
            end = 0
        if end >= len(self._player_batting[2022]):
            end = len(self._player_batting[2022]) - 1
        if start > end:
            end = start
        output = []
        current_batters = None
        if percentiles:
            current_batters = self._batting_percentiles[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        else:
            current_batters = self._player_batting[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        for i in range(len(current_batters)):
            if percentiles:
                output.append(BasicBatter(current_batters.iloc[i]['Name'], current_batters.iloc[i]['Team'], current_batters.iloc[i]['FLDPos'], self._percent(current_batters.iloc[i]['G']), self._percent(current_batters.iloc[i]['AB']), \
                    self._percent(current_batters.iloc[i]['PA']), self._percent(current_batters.iloc[i]['H']), self._percent(current_batters.iloc[i]['2B']), self._percent(current_batters.iloc[i]['3B']), self._percent(current_batters.iloc[i]['HR']), self._percent(current_batters.iloc[i]['BB']), \
                    self._percent(current_batters.iloc[i]['HBP']), self._percent(current_batters.iloc[i]['TB']), self._percent(current_batters.iloc[i]['R']), self._percent(current_batters.iloc[i]['RBI']), self._percent(current_batters.iloc[i]['SB']), self._percent(current_batters.iloc[i]['CS']), \
                    self._percent(current_batters.iloc[i]['SO']), self._percent(current_batters.iloc[i]['GDP']), self._percent(current_batters.iloc[i]['AVG']), self._percent(current_batters.iloc[i]['OBP']), self._percent(current_batters.iloc[i]['SLG']), self._percent(current_batters.iloc[i]['OPS'])))
            else:
                output.append(BasicBatter(current_batters.iloc[i]['Name'], current_batters.iloc[i]['Team'], current_batters.iloc[i]['FLDPos'], current_batters.iloc[i]['G'], current_batters.iloc[i]['AB'], \
                    current_batters.iloc[i]['PA'], current_batters.iloc[i]['H'], current_batters.iloc[i]['2B'], current_batters.iloc[i]['3B'], current_batters.iloc[i]['HR'], current_batters.iloc[i]['BB'], \
                    current_batters.iloc[i]['HBP'], current_batters.iloc[i]['TB'], current_batters.iloc[i]['R'], current_batters.iloc[i]['RBI'], current_batters.iloc[i]['SB'], current_batters.iloc[i]['CS'], \
                    current_batters.iloc[i]['SO'], current_batters.iloc[i]['GDP'], current_batters.iloc[i]['AVG'], current_batters.iloc[i]['OBP'], current_batters.iloc[i]['SLG'], current_batters.iloc[i]['OPS']))
        return output
    
    def get_player_batter_basic(self, name: str, percentiles: bool) -> list[PlayerBasicBatter]:
        output = []
        for i in range(2022, 2014, -1):
            if name not in list(self._player_batting[i]['Name']):
                continue
            if percentiles:
                batter = self._batting_percentiles[i][self._batting_percentiles[i]['Name'] == name]
                output.append(PlayerBasicBatter(batter.iloc[0]['Name'], batter.iloc[0]['Team'], i, batter.iloc[0]['FLDPos'], self._percent(batter.iloc[0]['G']), self._percent(batter.iloc[0]['AB']), \
                    self._percent(batter.iloc[0]['PA']), self._percent(batter.iloc[0]['H']), self._percent(batter.iloc[0]['2B']), self._percent(batter.iloc[0]['3B']), self._percent(batter.iloc[0]['HR']), self._percent(batter.iloc[0]['BB']), \
                    self._percent(batter.iloc[0]['HBP']), self._percent(batter.iloc[0]['TB']), self._percent(batter.iloc[0]['R']), self._percent(batter.iloc[0]['RBI']), self._percent(batter.iloc[0]['SB']), self._percent(batter.iloc[0]['CS']), \
                    self._percent(batter.iloc[0]['SO']), self._percent(batter.iloc[0]['GDP']), self._percent(batter.iloc[0]['AVG']), self._percent(batter.iloc[0]['OBP']), self._percent(batter.iloc[0]['SLG']), self._percent(batter.iloc[0]['OPS'])))
            else:
                batter = self._player_batting[i][self._player_batting[i]['Name'] == name]
                output.append(PlayerBasicBatter(batter.iloc[0]['Name'], batter.iloc[0]['Team'], i, batter.iloc[0]['FLDPos'], batter.iloc[0]['G'], batter.iloc[0]['AB'], \
                    batter.iloc[0]['PA'], batter.iloc[0]['H'], batter.iloc[0]['2B'], batter.iloc[0]['3B'], batter.iloc[0]['HR'], batter.iloc[0]['BB'], \
                    batter.iloc[0]['HBP'], batter.iloc[0]['TB'], batter.iloc[0]['R'], batter.iloc[0]['RBI'], batter.iloc[0]['SB'], batter.iloc[0]['CS'], \
                    batter.iloc[0]['SO'], batter.iloc[0]['GDP'], batter.iloc[0]['AVG'], batter.iloc[0]['OBP'], batter.iloc[0]['SLG'], batter.iloc[0]['OPS']))
        return output
    
    def get_advanced_batter_leaders(self, sortBy: str, start: int, end: int, ascend: bool, percentiles: bool) -> list[AdvancedBatter]:
        if start < 0:
            start = 0
        if start >= len(self._player_batting[2022]):
            start = len(self._player_batting[2022]) - 1
        if end < 0:
            end = 0
        if end >= len(self._player_batting[2022]):
            end = len(self._player_batting[2022]) - 1
        if start > end:
            end = start
        output = []
        current_batters = None
        if percentiles:
            current_batters = self._batting_percentiles[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        else:
            current_batters = self._player_batting[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        for i in range(len(current_batters)):
            if percentiles:
                output.append(AdvancedBatter(current_batters.iloc[i]['Name'], current_batters.iloc[i]['Team'], current_batters.iloc[i]['FLDPos'], self._percent(current_batters.iloc[i]['G']), self._percent(current_batters.iloc[i]['PA']), \
                self._percent(current_batters.iloc[i]['AVG']), self._percent(current_batters.iloc[i]['OBP']), self._percent(current_batters.iloc[i]['SLG']), self._percent(current_batters.iloc[i]['OPS']), self._percent(current_batters.iloc[i]['wRC+']), self._percent(current_batters.iloc[i]['wOBA']), \
                self._percent(current_batters.iloc[i]['ISO']), self._percent(current_batters.iloc[i]['BABIP']), self._percent(current_batters.iloc[i]['Clutch']), self._percent(current_batters.iloc[i]['BB%']), self._percent(current_batters.iloc[i]['K%']), self._percent(current_batters.iloc[i]['xBA']), \
                self._percent(current_batters.iloc[i]['Bat']), self._percent(current_batters.iloc[i]['Fld']), self._percent(current_batters.iloc[i]['WAR']), self._percent(current_batters.iloc[i]['bWAR']), self._percent(current_batters.iloc[i]['aWAR']), self._percent(current_batters.iloc[i]['WPA'])))
            else:
                output.append(AdvancedBatter(current_batters.iloc[i]['Name'], current_batters.iloc[i]['Team'], current_batters.iloc[i]['FLDPos'], current_batters.iloc[i]['G'], current_batters.iloc[i]['PA'], \
                    current_batters.iloc[i]['AVG'], current_batters.iloc[i]['OBP'], current_batters.iloc[i]['SLG'], current_batters.iloc[i]['OPS'], current_batters.iloc[i]['wRC+'], current_batters.iloc[i]['wOBA'], \
                    current_batters.iloc[i]['ISO'], current_batters.iloc[i]['BABIP'], current_batters.iloc[i]['Clutch'], current_batters.iloc[i]['BB%'], round(current_batters.iloc[i]['K%'], 3), current_batters.iloc[i]['xBA'], \
                    current_batters.iloc[i]['Bat'], current_batters.iloc[i]['Fld'], current_batters.iloc[i]['WAR'], current_batters.iloc[i]['bWAR'], current_batters.iloc[i]['aWAR'], current_batters.iloc[i]['WPA']))
        return output
    
    def get_player_batter_advanced(self, name: str, percentiles: bool) -> list[PlayerAdvancedBatter]:
        output = []
        for i in range(2022, 2014, -1):
            if name not in list(self._player_batting[i]['Name']):
                continue
            if percentiles:
                batter = self._batting_percentiles[i][self._batting_percentiles[i]['Name'] == name]
                output.append(PlayerAdvancedBatter(batter.iloc[0]['Name'], batter.iloc[0]['Team'], i, batter.iloc[0]['FLDPos'], self._percent(batter.iloc[0]['G']), self._percent(batter.iloc[0]['PA']), \
                self._percent(batter.iloc[0]['AVG']), self._percent(batter.iloc[0]['OBP']), self._percent(batter.iloc[0]['SLG']), self._percent(batter.iloc[0]['OPS']), self._percent(batter.iloc[0]['wRC+']), self._percent(batter.iloc[0]['wOBA']), \
                self._percent(batter.iloc[0]['ISO']), self._percent(batter.iloc[0]['BABIP']), self._percent(batter.iloc[0]['Clutch']), self._percent(batter.iloc[0]['BB%']), self._percent(batter.iloc[0]['K%']), self._percent(batter.iloc[0]['xBA']), \
                self._percent(batter.iloc[0]['Bat']), self._percent(batter.iloc[0]['Fld']), self._percent(batter.iloc[0]['WAR']), self._percent(batter.iloc[0]['bWAR']), self._percent(batter.iloc[0]['aWAR']), self._percent(batter.iloc[0]['WPA'])))
            else:
                batter = self._player_batting[i][self._player_batting[i]['Name'] == name]
                output.append(PlayerAdvancedBatter(batter.iloc[0]['Name'], batter.iloc[0]['Team'], i, batter.iloc[0]['FLDPos'], batter.iloc[0]['G'], batter.iloc[0]['PA'], \
                    batter.iloc[0]['AVG'], batter.iloc[0]['OBP'], batter.iloc[0]['SLG'], batter.iloc[0]['OPS'], batter.iloc[0]['wRC+'], batter.iloc[0]['wOBA'], \
                    batter.iloc[0]['ISO'], batter.iloc[0]['BABIP'], batter.iloc[0]['Clutch'], batter.iloc[0]['BB%'], round(batter.iloc[0]['K%'], 3), batter.iloc[0]['xBA'], \
                    batter.iloc[0]['Bat'], batter.iloc[0]['Fld'], batter.iloc[0]['WAR'], batter.iloc[0]['bWAR'], batter.iloc[0]['aWAR'], batter.iloc[0]['WPA']))
        return output
    
    def get_basic_pitcher_leaders(self, sortBy: str, start: int, end: int, ascend: bool, percentiles: bool) -> list[BasicPitcher]:
        if start < 0:
            start = 0
        if start >= len(self._player_pitching[2022]):
            start = len(self._player_pitching[2022]) - 1
        if end < 0:
            end = 0
        if end >= len(self._player_pitching[2022]):
            end = len(self._player_pitching[2022]) - 1
        if start > end:
            end = start
        output = []
        current_pitchers = None
        if percentiles:
            current_pitchers = self._pitching_percentiles[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        else:
            current_pitchers = self._player_pitching[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        for i in range(len(current_pitchers)):
            if percentiles:
                output.append(BasicPitcher(current_pitchers.iloc[i]['Name'], current_pitchers.iloc[i]['Team'], self._percent(current_pitchers.iloc[i]['G']), self._percent(current_pitchers.iloc[i]['GS']), self._percent(current_pitchers.iloc[i]['IP']), \
                    self._percent(current_pitchers.iloc[i]['W']), self._percent(current_pitchers.iloc[i]['L']), self._percent(current_pitchers.iloc[i]['CG']), self._percent(current_pitchers.iloc[i]['SV']), self._percent(current_pitchers.iloc[i]['BS']), self._percent(current_pitchers.iloc[i]['ER']), \
                    self._percent(current_pitchers.iloc[i]['H']), self._percent(current_pitchers.iloc[i]['BB']), self._percent(current_pitchers.iloc[i]['SO']), self._percent(current_pitchers.iloc[i]['ERA']), self._percent(current_pitchers.iloc[i]['WHIP'])))
            else:
                output.append(BasicPitcher(current_pitchers.iloc[i]['Name'], current_pitchers.iloc[i]['Team'], current_pitchers.iloc[i]['G'], current_pitchers.iloc[i]['GS'], current_pitchers.iloc[i]['IP'], \
                    current_pitchers.iloc[i]['W'], current_pitchers.iloc[i]['L'], current_pitchers.iloc[i]['CG'], current_pitchers.iloc[i]['SV'], current_pitchers.iloc[i]['BS'], current_pitchers.iloc[i]['ER'], \
                    current_pitchers.iloc[i]['H'], current_pitchers.iloc[i]['BB'], current_pitchers.iloc[i]['SO'], current_pitchers.iloc[i]['ERA'], current_pitchers.iloc[i]['WHIP']))
        return output
    
    def get_player_pitcher_basic(self, name: str, percentiles: bool) -> list[PlayerBasicPitcher]:
        output = []
        for i in range(2022, 2014, -1):
            if name not in list(self._player_pitching[i]['Name']):
                continue
            if percentiles:
                pitcher = self._pitching_percentiles[i][self._pitching_percentiles[i]['Name'] == name]
                output.append(PlayerBasicPitcher(pitcher.iloc[0]['Name'], pitcher.iloc[0]['Team'], i, self._percent(pitcher.iloc[0]['G']), self._percent(pitcher.iloc[0]['GS']), self._percent(pitcher.iloc[0]['IP']), \
                    self._percent(pitcher.iloc[0]['W']), self._percent(pitcher.iloc[0]['L']), self._percent(pitcher.iloc[0]['CG']), self._percent(pitcher.iloc[0]['SV']), self._percent(pitcher.iloc[0]['BS']), self._percent(pitcher.iloc[0]['ER']), \
                    self._percent(pitcher.iloc[0]['H']), self._percent(pitcher.iloc[0]['BB']), self._percent(pitcher.iloc[0]['SO']), self._percent(pitcher.iloc[0]['ERA']), self._percent(pitcher.iloc[0]['WHIP'])))
            else:
                pitcher = self._player_pitching[i][self._player_pitching[i]['Name'] == name]
                output.append(PlayerBasicPitcher(pitcher.iloc[0]['Name'], pitcher.iloc[0]['Team'], i, pitcher.iloc[0]['G'], pitcher.iloc[0]['GS'], pitcher.iloc[0]['IP'], \
                    pitcher.iloc[0]['W'], pitcher.iloc[0]['L'], pitcher.iloc[0]['CG'], pitcher.iloc[0]['SV'], pitcher.iloc[0]['BS'], pitcher.iloc[0]['ER'], \
                    pitcher.iloc[0]['H'], pitcher.iloc[0]['BB'], pitcher.iloc[0]['SO'], pitcher.iloc[0]['ERA'], pitcher.iloc[0]['WHIP']))
        return output
    
    def get_advanced_pitcher_leaders(self, sortBy: str, start: int, end: int, ascend: bool, percentiles: bool) -> list[AdvancedPitcher]:
        if start < 0:
            start = 0
        if start >= len(self._player_pitching[2022]):
            start = len(self._player_pitching[2022]) - 1
        if end < 0:
            end = 0
        if end >= len(self._player_pitching[2022]):
            end = len(self._player_pitching[2022]) - 1
        if start > end:
            end = start
        output = []
        current_pitchers = None
        if percentiles:
            current_pitchers = self._pitching_percentiles[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        else:
            current_pitchers = self._player_pitching[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        for i in range(len(current_pitchers)):
            if percentiles:
                output.append(AdvancedPitcher(current_pitchers.iloc[i]['Name'], current_pitchers.iloc[i]['Team'], self._percent(current_pitchers.iloc[i]['G']), self._percent(current_pitchers.iloc[i]['IP']), self._percent(current_pitchers.iloc[i]['K/9']), \
                    self._percent(current_pitchers.iloc[i]['BB/9']), self._percent(current_pitchers.iloc[i]['HR/9']), self._percent(current_pitchers.iloc[i]['K/BB']), self._percent(current_pitchers.iloc[i]['AVG']), self._percent(current_pitchers.iloc[i]['Clutch']), \
                    self._percent(current_pitchers.iloc[i]['BABIP']), self._percent(current_pitchers.iloc[i]['LOB%']), self._percent(current_pitchers.iloc[i]['WHIP']), self._percent(current_pitchers.iloc[i]['ERA']), self._percent(current_pitchers.iloc[i]['RS/9']), \
                    self._percent(current_pitchers.iloc[i]['FIP']), self._percent(current_pitchers.iloc[i]['xFIP']), self._percent(current_pitchers.iloc[i]['SIERA']), self._percent(current_pitchers.iloc[i]['WAR']), self._percent(current_pitchers.iloc[i]['bWAR']), \
                    self._percent(current_pitchers.iloc[i]['aWAR']), self._percent(current_pitchers.iloc[i]['WPA'])))
            else:
                output.append(AdvancedPitcher(current_pitchers.iloc[i]['Name'], current_pitchers.iloc[i]['Team'], current_pitchers.iloc[i]['G'], current_pitchers.iloc[i]['IP'], current_pitchers.iloc[i]['K/9'], \
                    current_pitchers.iloc[i]['BB/9'], current_pitchers.iloc[i]['HR/9'], current_pitchers.iloc[i]['K/BB'], current_pitchers.iloc[i]['AVG'], current_pitchers.iloc[i]['Clutch'], \
                    current_pitchers.iloc[i]['BABIP'], round(current_pitchers.iloc[i]['LOB%'], 3), current_pitchers.iloc[i]['WHIP'], current_pitchers.iloc[i]['ERA'], current_pitchers.iloc[i]['RS/9'], \
                    current_pitchers.iloc[i]['FIP'], current_pitchers.iloc[i]['xFIP'], current_pitchers.iloc[i]['SIERA'], current_pitchers.iloc[i]['WAR'], current_pitchers.iloc[i]['bWAR'], \
                    current_pitchers.iloc[i]['aWAR'], current_pitchers.iloc[i]['WPA']))
        return output
    
    def get_player_pitcher_advanced(self, name: str, percentiles: bool) -> list[PlayerAdvancedPitcher]:
        output = []
        for i in range(2022, 2014, -1):
            if name not in list(self._player_pitching[i]['Name']):
                continue
            if percentiles:
                pitcher = self._pitching_percentiles[i][self._pitching_percentiles[i]['Name'] == name]
                output.append(PlayerAdvancedPitcher(pitcher.iloc[0]['Name'], pitcher.iloc[0]['Team'], i, self._percent(pitcher.iloc[0]['G']), self._percent(pitcher.iloc[0]['IP']), self._percent(pitcher.iloc[0]['K/9']), \
                    self._percent(pitcher.iloc[0]['BB/9']), self._percent(pitcher.iloc[0]['HR/9']), self._percent(pitcher.iloc[0]['K/BB']), self._percent(pitcher.iloc[0]['AVG']), self._percent(pitcher.iloc[0]['Clutch']), \
                    self._percent(pitcher.iloc[0]['BABIP']), self._percent(pitcher.iloc[0]['LOB%']), self._percent(pitcher.iloc[0]['WHIP']), self._percent(pitcher.iloc[0]['ERA']), self._percent(pitcher.iloc[0]['RS/9']), \
                    self._percent(pitcher.iloc[0]['FIP']), self._percent(pitcher.iloc[0]['xFIP']), self._percent(pitcher.iloc[0]['SIERA']), self._percent(pitcher.iloc[0]['WAR']), self._percent(pitcher.iloc[0]['bWAR']), \
                    self._percent(pitcher.iloc[0]['aWAR']), self._percent(pitcher.iloc[0]['WPA'])))
            else:
                pitcher = self._player_pitching[i][self._player_pitching[i]['Name'] == name]
                output.append(PlayerAdvancedPitcher(pitcher.iloc[0]['Name'], pitcher.iloc[0]['Team'], i, pitcher.iloc[0]['G'], pitcher.iloc[0]['IP'], pitcher.iloc[0]['K/9'], \
                    pitcher.iloc[0]['BB/9'], pitcher.iloc[0]['HR/9'], pitcher.iloc[0]['K/BB'], pitcher.iloc[0]['AVG'], pitcher.iloc[0]['Clutch'], \
                    pitcher.iloc[0]['BABIP'], round(pitcher.iloc[0]['LOB%'], 3), pitcher.iloc[0]['WHIP'], pitcher.iloc[0]['ERA'], pitcher.iloc[0]['RS/9'], \
                    pitcher.iloc[0]['FIP'], pitcher.iloc[0]['xFIP'], pitcher.iloc[0]['SIERA'], pitcher.iloc[0]['WAR'], pitcher.iloc[0]['bWAR'], \
                    pitcher.iloc[0]['aWAR'], pitcher.iloc[0]['WPA']))
        return output
    
    def get_statcast_batter_leaders(self, sortBy: str, start: int, end: int, ascend: bool, percentiles: bool) -> list[StatcastHitter]:
        if start < 0:
            start = 0
        if start >= len(self._player_batting[2022]):
            start = len(self._player_batting[2022]) - 1
        if end < 0:
            end = 0
        if end >= len(self._player_batting[2022]):
            end = len(self._player_batting[2022]) - 1
        if start > end:
            end = start
        output = []
        current_batters = None
        if percentiles:
            current_batters = self._batting_percentiles[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        else:
            current_batters = self._player_batting[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        for i in range(len(current_batters)):
            if percentiles:
                output.append(StatcastHitter(current_batters.iloc[i]['Name'], current_batters.iloc[i]['Team'], current_batters.iloc[i]['FLDPos'], self._percent(current_batters.iloc[i]['PA']), self._percent(current_batters.iloc[i]['Barrel%']), \
                    self._percent(current_batters.iloc[i]['HardHit%']), self._percent(current_batters.iloc[i]['LD%']), self._percent(current_batters.iloc[i]['GB%']), self._percent(current_batters.iloc[i]['FB%']), self._percent(current_batters.iloc[i]['O-Swing%']), \
                    self._percent(current_batters.iloc[i]['Z-Swing%']), self._percent(current_batters.iloc[i]['Contact%']), self._percent(current_batters.iloc[i]['Pull%']), self._percent(current_batters.iloc[i]['Cent%']), self._percent(current_batters.iloc[i]['Oppo%']), \
                    self._percent(current_batters.iloc[i]['SwStr%']), self._percent(current_batters.iloc[i]['EV']), self._percent(current_batters.iloc[i]['maxEV']), self._percent(current_batters.iloc[i]['LA']), self._percent(current_batters.iloc[i]['xBA']), \
                    self._percent(current_batters.iloc[i]['xSLG']), self._percent(current_batters.iloc[i]['xwOBA']), self._percent(current_batters.iloc[i]['BABIP'])))
            else:
                output.append(StatcastHitter(current_batters.iloc[i]['Name'], current_batters.iloc[i]['Team'], current_batters.iloc[i]['FLDPos'], current_batters.iloc[i]['PA'], round(current_batters.iloc[i]['Barrel%'], 3), \
                    round(current_batters.iloc[i]['HardHit%'], 3), round(current_batters.iloc[i]['LD%'], 3), round(current_batters.iloc[i]['GB%'], 3), round(current_batters.iloc[i]['FB%'], 3), round(current_batters.iloc[i]['O-Swing%'], 3), \
                    round(current_batters.iloc[i]['Z-Swing%'], 3), round(current_batters.iloc[i]['Contact%'], 3), round(current_batters.iloc[i]['Pull%'], 3), round(current_batters.iloc[i]['Cent%'], 3), round(current_batters.iloc[i]['Oppo%'], 3), \
                    round(current_batters.iloc[i]['SwStr%'], 3), current_batters.iloc[i]['EV'], current_batters.iloc[i]['maxEV'], current_batters.iloc[i]['LA'], current_batters.iloc[i]['xBA'], \
                    current_batters.iloc[i]['xSLG'], current_batters.iloc[i]['xwOBA'], current_batters.iloc[i]['BABIP']))
        return output
    
    def get_player_batter_statcast(self, name: str, percentiles: bool) -> list[PlayerStatcastHitter]:
        output = []
        for i in range(2022, 2014, -1):
            if name not in list(self._player_batting[i]['Name']):
                continue
            if percentiles:
                batter = self._batting_percentiles[i][self._batting_percentiles[i]['Name'] == name]
                output.append(PlayerStatcastHitter(batter.iloc[0]['Name'], batter.iloc[0]['Team'], i, batter.iloc[0]['FLDPos'], self._percent(batter.iloc[0]['PA']), self._percent(batter.iloc[0]['Barrel%']), \
                    self._percent(batter.iloc[0]['HardHit%']), self._percent(batter.iloc[0]['LD%']), self._percent(batter.iloc[0]['GB%']), self._percent(batter.iloc[0]['FB%']), self._percent(batter.iloc[0]['O-Swing%']), \
                    self._percent(batter.iloc[0]['Z-Swing%']), self._percent(batter.iloc[0]['Contact%']), self._percent(batter.iloc[0]['Pull%']), self._percent(batter.iloc[0]['Cent%']), self._percent(batter.iloc[0]['Oppo%']), \
                    self._percent(batter.iloc[0]['SwStr%']), self._percent(batter.iloc[0]['EV']), self._percent(batter.iloc[0]['maxEV']), self._percent(batter.iloc[0]['LA']), self._percent(batter.iloc[0]['xBA']), \
                    self._percent(batter.iloc[0]['xSLG']), self._percent(batter.iloc[0]['xwOBA']), self._percent(batter.iloc[0]['BABIP'])))
            else:
                batter = self._player_batting[i][self._player_batting[i]['Name'] == name]
                output.append(PlayerStatcastHitter(batter.iloc[0]['Name'], batter.iloc[0]['Team'], i, batter.iloc[0]['FLDPos'], batter.iloc[0]['PA'], round(batter.iloc[0]['Barrel%'], 3), \
                    round(batter.iloc[0]['HardHit%'], 3), round(batter.iloc[0]['LD%'], 3), round(batter.iloc[0]['GB%'], 3), round(batter.iloc[0]['FB%'], 3), round(batter.iloc[0]['O-Swing%']), \
                    round(batter.iloc[0]['Z-Swing%'], 3), round(batter.iloc[0]['Contact%'], 3), round(batter.iloc[0]['Pull%'], 3), round(batter.iloc[0]['Cent%'], 3), round(batter.iloc[0]['Oppo%'], 3), \
                    round(batter.iloc[0]['SwStr%'], 3), batter.iloc[0]['EV'], batter.iloc[0]['maxEV'], batter.iloc[0]['LA'], batter.iloc[0]['xBA'], \
                    batter.iloc[0]['xSLG'], batter.iloc[0]['xwOBA'], batter.iloc[0]['BABIP']))
        return output
    
    def get_statcast_fielder_leaders(self, sortBy: str, start: int, end: int, ascend: bool, percentiles: bool) -> list[StatcastFielder]:
        if start < 0:
            start = 0
        if start >= len(self._statcast_fielder[2022]):
            start = len(self._statcast_fielder[2022]) - 1
        if end < 0:
            end = 0
        if end >= len(self._statcast_fielder[2022]):
            end = len(self._statcast_fielder[2022]) - 1
        if start > end:
            end = start
        output = []
        current_fielders = None
        if percentiles:
            current_fielders = self._fielding_statcast_percentiles[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        else:
            current_fielders = self._statcast_fielder[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        for i in range(len(current_fielders)):
            name = current_fielders.iloc[i]['first_name'] + ' ' + current_fielders.iloc[i]['last_name']
            if percentiles:
                output.append(StatcastFielder(name, current_fielders.iloc[i]['team'], current_fielders.iloc[i]['primary_pos_formatted'], self._percent(current_fielders.iloc[i]['outs_above_average_x']), self._percent(current_fielders.iloc[i]['outs_above_average_infront']), \
                self._percent(current_fielders.iloc[i]['outs_above_average_lateral_toward3bline']), self._percent(current_fielders.iloc[i]['outs_above_average_lateral_toward1bline']), self._percent(current_fielders.iloc[i]['outs_above_average_behind']), \
                self._percent(current_fielders.iloc[i]['actual_success_rate_formatted']), self._percent(current_fielders.iloc[i]['adj_estimated_success_rate_formatted']), self._percent(current_fielders.iloc[i]['fielding_runs_prevented']), \
                self._percent(current_fielders.iloc[i]['rel_league_reaction_distance']), self._percent(current_fielders.iloc[i]['rel_league_burst_distance']), self._percent(current_fielders.iloc[i]['rel_league_routing_distance']), \
                self._percent(current_fielders.iloc[i]['f_bootup_distance']), self._percent(current_fielders.iloc[i]['sprint_speed']), self._percent(current_fielders.iloc[i]['hp_to_1b']), self._percent(current_fielders.iloc[i]['bolts'])))
            else:
                output.append(StatcastFielder(name, current_fielders.iloc[i]['team'], current_fielders.iloc[i]['primary_pos_formatted'], current_fielders.iloc[i]['outs_above_average_x'], current_fielders.iloc[i]['outs_above_average_infront'], \
                current_fielders.iloc[i]['outs_above_average_lateral_toward3bline'], current_fielders.iloc[i]['outs_above_average_lateral_toward1bline'], current_fielders.iloc[i]['outs_above_average_behind'], \
                current_fielders.iloc[i]['actual_success_rate_formatted'], current_fielders.iloc[i]['adj_estimated_success_rate_formatted'], current_fielders.iloc[i]['fielding_runs_prevented'], \
                current_fielders.iloc[i]['rel_league_reaction_distance'], current_fielders.iloc[i]['rel_league_burst_distance'], current_fielders.iloc[i]['rel_league_routing_distance'], \
                current_fielders.iloc[i]['f_bootup_distance'], current_fielders.iloc[i]['sprint_speed'], current_fielders.iloc[i]['hp_to_1b'], current_fielders.iloc[i]['bolts']))
        return output
    
    def get_player_fielder_statcast(self, name: str, percentiles: bool) -> list[PlayerStatcastFielder]:
        output = []
        split = name.split(' ')
        for i in range(2022, 2014, -1):
            if split[0] not in list(self._statcast_fielder[i]['first_name']) or split[1] not in list(self._statcast_fielder[i]['last_name']) or split[1] not in list(self._statcast_fielder[i][self._statcast_fielder[i]['first_name'] == split[0]]['last_name']):
                continue
            if percentiles:
                fielder = self._fielding_statcast_percentiles[i][(self._fielding_statcast_percentiles[i]['first_name'] == split[0]) & (self._fielding_statcast_percentiles[i]['last_name'] == split[1])]
                output.append(PlayerStatcastFielder(name, fielder.iloc[0]['team'], i, fielder.iloc[0]['primary_pos_formatted'], self._percent(fielder.iloc[0]['outs_above_average_x']), self._percent(fielder.iloc[0]['outs_above_average_infront']), \
                self._percent(fielder.iloc[0]['outs_above_average_lateral_toward3bline']), self._percent(fielder.iloc[0]['outs_above_average_lateral_toward1bline']), self._percent(fielder.iloc[0]['outs_above_average_behind']), \
                self._percent(fielder.iloc[0]['actual_success_rate_formatted']), self._percent(fielder.iloc[0]['adj_estimated_success_rate_formatted']), self._percent(fielder.iloc[0]['fielding_runs_prevented']), \
                self._percent(fielder.iloc[0]['rel_league_reaction_distance']), self._percent(fielder.iloc[0]['rel_league_burst_distance']), self._percent(fielder.iloc[0]['rel_league_routing_distance']), \
                self._percent(fielder.iloc[0]['f_bootup_distance']), self._percent(fielder.iloc[0]['sprint_speed']), self._percent(fielder.iloc[0]['hp_to_1b']), self._percent(fielder.iloc[0]['bolts'])))
            else:
                fielder = self._statcast_fielder[i][(self._statcast_fielder[i]['first_name'] == split[0]) & (self._statcast_fielder[i]['last_name'] == split[1])]
                output.append(PlayerStatcastFielder(name, fielder.iloc[0]['team'], i, fielder.iloc[0]['primary_pos_formatted'], fielder.iloc[0]['outs_above_average_x'], fielder.iloc[0]['outs_above_average_infront'], \
                    fielder.iloc[0]['outs_above_average_lateral_toward3bline'], fielder.iloc[0]['outs_above_average_lateral_toward1bline'], fielder.iloc[0]['outs_above_average_behind'], \
                    fielder.iloc[0]['actual_success_rate_formatted'], fielder.iloc[0]['adj_estimated_success_rate_formatted'], fielder.iloc[0]['fielding_runs_prevented'], \
                    fielder.iloc[0]['rel_league_reaction_distance'], fielder.iloc[0]['rel_league_burst_distance'], fielder.iloc[0]['rel_league_routing_distance'], \
                    fielder.iloc[0]['f_bootup_distance'], fielder.iloc[0]['sprint_speed'], fielder.iloc[0]['hp_to_1b'], fielder.iloc[0]['bolts']))
        return output
    
    def get_statcast_pitcher_leaders(self, sortBy: str, start: int, end: int, ascend: bool, percentiles: bool) -> list[StatcastPitcher]:
        if start < 0:
            start = 0
        if start >= len(self._player_pitching[2022]):
            start = len(self._player_pitching[2022]) - 1
        if end < 0:
            end = 0
        if end >= len(self._player_pitching[2022]):
            end = len(self._player_pitching[2022]) - 1
        if start > end:
            end = start
        output = []
        current_pitchers = None
        if percentiles:
            current_pitchers = self._pitching_percentiles[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        else:
            current_pitchers = self._player_pitching[2022].sort_values(by=sortBy, ascending=ascend).iloc[start:end]
        for i in range(len(current_pitchers)):
            if percentiles:
                output.append(StatcastPitcher(current_pitchers.iloc[i]['Name'], current_pitchers.iloc[i]['Team'], self._percent(current_pitchers.iloc[i]['G']), self._percent(current_pitchers.iloc[i]['IP']), self._percent(current_pitchers.iloc[i]['HardHit%']), self._percent(current_pitchers.iloc[i]['Barrel%']), \
                self._percent(current_pitchers.iloc[i]['K%']), self._percent(current_pitchers.iloc[i]['BB%']), self._percent(current_pitchers.iloc[i]['LD%']), self._percent(current_pitchers.iloc[i]['GB%']), self._percent(current_pitchers.iloc[i]['FB%']), self._percent(current_pitchers.iloc[i]['K/BB']), \
                self._percent(current_pitchers.iloc[i]['GB/FB']), self._percent(current_pitchers.iloc[i]['HR/FB']), self._percent(current_pitchers.iloc[i]['Soft%']), self._percent(current_pitchers.iloc[i]['Med%']), self._percent(current_pitchers.iloc[i]['EV']), \
                self._percent(current_pitchers.iloc[i]['maxEV']), self._percent(current_pitchers.iloc[i]['NumPitches']), self._percent(current_pitchers.iloc[i]['MaxVelo']), self._percent(current_pitchers.iloc[i]['AvrSpin']), self._percent(current_pitchers.iloc[i]['est_ba']), \
                self._percent(current_pitchers.iloc[i]['est_slg']), self._percent(current_pitchers.iloc[i]['est_woba']), self._percent(current_pitchers.iloc[i]['xERA'])))
            else:
                output.append(StatcastPitcher(current_pitchers.iloc[i]['Name'], current_pitchers.iloc[i]['Team'], current_pitchers.iloc[i]['G'], current_pitchers.iloc[i]['IP'], round(current_pitchers.iloc[i]['HardHit%'], 3), round(current_pitchers.iloc[i]['Barrel%'], 3), \
                round(current_pitchers.iloc[i]['K%'], 3), round(current_pitchers.iloc[i]['BB%'], 3), round(current_pitchers.iloc[i]['LD%'], 3), round(current_pitchers.iloc[i]['GB%'], 3), round(current_pitchers.iloc[i]['FB%'], 3), round(current_pitchers.iloc[i]['K/BB'], 3), \
                round(current_pitchers.iloc[i]['GB/FB'], 3), round(current_pitchers.iloc[i]['HR/FB'], 3), round(current_pitchers.iloc[i]['Soft%'], 3), round(current_pitchers.iloc[i]['Med%'], 3), current_pitchers.iloc[i]['EV'], \
                current_pitchers.iloc[i]['maxEV'], current_pitchers.iloc[i]['NumPitches'], current_pitchers.iloc[i]['MaxVelo'], round(current_pitchers.iloc[i]['AvrSpin'], 1), current_pitchers.iloc[i]['est_ba'], \
                current_pitchers.iloc[i]['est_slg'], current_pitchers.iloc[i]['est_woba'], current_pitchers.iloc[i]['xERA']))
        return output
    
    def get_player_pitcher_statcast(self, name: str, percentiles: bool) -> list[PlayerStatcastPitcher]:
        output = []
        for i in range(2022, 2014, -1):
            if name not in list(self._player_pitching[i]['Name']):
                continue
            if percentiles:
                pitcher = self._pitching_percentiles[i][self._pitching_percentiles[i]['Name'] == name]
                output.append(PlayerStatcastPitcher(pitcher.iloc[0]['Name'], pitcher.iloc[0]['Team'], i, self._percent(pitcher.iloc[0]['G']), self._percent(pitcher.iloc[0]['IP']), self._percent(pitcher.iloc[0]['HardHit%']), self._percent(pitcher.iloc[0]['Barrel%']), \
                self._percent(pitcher.iloc[0]['K%']), self._percent(pitcher.iloc[0]['BB%']), self._percent(pitcher.iloc[0]['LD%']), self._percent(pitcher.iloc[0]['GB%']), self._percent(pitcher.iloc[0]['FB%']), self._percent(pitcher.iloc[0]['K/BB']), \
                self._percent(pitcher.iloc[0]['GB/FB']), self._percent(pitcher.iloc[0]['HR/FB']), self._percent(pitcher.iloc[0]['Soft%']), self._percent(pitcher.iloc[0]['Med%']), self._percent(pitcher.iloc[0]['EV']), \
                self._percent(pitcher.iloc[0]['maxEV']), self._percent(pitcher.iloc[0]['NumPitches']), self._percent(pitcher.iloc[0]['MaxVelo']), self._percent(pitcher.iloc[0]['AvrSpin']), self._percent(pitcher.iloc[0]['est_ba']), \
                self._percent(pitcher.iloc[0]['est_slg']), self._percent(pitcher.iloc[0]['est_woba']), self._percent(pitcher.iloc[0]['xERA'])))
            else:
                pitcher = self._player_pitching[i][self._player_pitching[i]['Name'] == name]
                output.append(PlayerStatcastPitcher(pitcher.iloc[0]['Name'], pitcher.iloc[0]['Team'], i, pitcher.iloc[0]['G'], pitcher.iloc[0]['IP'], round(pitcher.iloc[0]['HardHit%'], 3), round(pitcher.iloc[0]['Barrel%'], 3), \
                    round(pitcher.iloc[0]['K%'], 3), round(pitcher.iloc[0]['BB%'], 3), round(pitcher.iloc[0]['LD%'], 3), round(pitcher.iloc[0]['GB%'], 3), round(pitcher.iloc[0]['FB%'], 3), round(pitcher.iloc[0]['K/BB'], 3), \
                    round(pitcher.iloc[0]['GB/FB'], 3), round(pitcher.iloc[0]['HR/FB'], 3), round(pitcher.iloc[0]['Soft%'], 3), round(pitcher.iloc[0]['Med%'], 3), pitcher.iloc[0]['EV'], \
                    pitcher.iloc[0]['maxEV'], pitcher.iloc[0]['NumPitches'], pitcher.iloc[0]['MaxVelo'], round(pitcher.iloc[0]['AvrSpin'], 1), pitcher.iloc[0]['est_ba'], \
                    pitcher.iloc[0]['est_slg'], pitcher.iloc[0]['est_woba'], pitcher.iloc[0]['xERA']))
        return output
    
    def get_list_player_names(self, year: int) -> list[str]:
        return list(self._player_batting[year]['Name']) + list(self._player_pitching[year]['Name'])
    
    def is_hitter(self, name: str) -> bool:
        return name in list(self._player_batting[2022]['Name'])
    
    def get_full_standings(self, division: int) -> list[TeamFull]:
        output = []
        for i in range(len(self._standing[2022][division])):
            abbr = team_abbr[self._standing[2022][division].iloc[i]['Tm']]
            rf = self._team_hitting[2022][self._team_hitting[2022]['Team'] == abbr].iloc[0]['R']
            ra = self._team_pitching[2022][self._team_pitching[2022]['Team'] == abbr].iloc[0]['R']
            pyth_w = self._calc_pyth_wins(rf, ra)
            home_w = self._get_home_w(abbr)
            home_l = self._get_home_l(abbr)
            road_w = self._standing[2022][division].iloc[i]['W'] - home_w
            road_l = self._standing[2022][division].iloc[i]['L'] - home_l
            over_500_w = self._get_over_500_w(abbr)
            over_500_l = self._get_over_500_l(abbr)
            under_500_w = self._standing[2022][division].iloc[i]['W'] - over_500_w
            under_500_l = self._standing[2022][division].iloc[i]['L'] - over_500_l
            output.append(TeamFull(self._standing[2022][division].iloc[i]['Tm'], self._standing[2022][division].iloc[i]['W'], self._standing[2022][division].iloc[i]['L'], self._standing[2022][division].iloc[i]['W-L%'], self._standing[2022][division].iloc[i]['GB'], \
                self._get_streak(self._standing[2022][division].iloc[i]['Tm']), (rf-ra), rf, round((rf/162), 2), ra, round((ra/162), 2), pyth_w, (162-pyth_w), str(home_w) + '-' + str(home_l), str(road_w) + '-' + str(road_l), \
                str(over_500_w) + '-' + str(over_500_l), str(under_500_w) + '-' + str(under_500_l), rf, ra, (rf-ra), self._standing[2022][division].iloc[i]['W'], self._standing[2022][division].iloc[i]['L']))
        return output
    
    def get_wildcard(self, west: list[TeamFull], central: list[TeamFull], east: list[TeamFull]) -> list[TeamFull]:
        output = []
        westCount = 1
        centralCount = 1
        eastCount = 1
        temp_GB = 0
        while westCount < 5 or eastCount < 5 or centralCount < 5:
            if self._get_wins(westCount, west) > self._get_wins(centralCount, central) and self._get_wins(westCount, west) > self._get_wins(eastCount, east):
                output.append(west[westCount])
                westCount += 1
            elif self._get_wins(centralCount, central) > self._get_wins(westCount, west) and self._get_wins(centralCount, central) > self._get_wins(eastCount, east):
                output.append(central[centralCount])
                centralCount += 1
            else:
                if self._get_wins(eastCount, east) == -1:
                    output.append(west[westCount])
                    westCount += 1
                else:
                    output.append(east[eastCount])
                    eastCount += 1
            if (len(output) == 3):
                output[2] = TeamFull(output[2].Team, output[2].W, output[2].L, output[2].WPercent, '--', output[2].Strk, output[2].Diff, output[2].Rf, output[2].RfG, output[2].Ra, output[2].RaG, output[2].PythW, output[2].PythL, 
                                    output[2].HomeWL, output[2].RoadWL, output[2].Over500WL, output[2].Under500WL, output[2].ProjRf, output[2].ProjRa, output[2].ProjDiff, output[2].ProjW, output[2].ProjL)
                output[1] = TeamFull(output[1].Team, output[1].W, output[1].L, output[1].WPercent, '+' + str(output[1].W - output[2].W), output[1].Strk, output[1].Diff, output[1].Rf, output[1].RfG, output[1].Ra, output[1].RaG, output[1].PythW, output[1].PythL, 
                                    output[1].HomeWL, output[1].RoadWL, output[1].Over500WL, output[1].Under500WL, output[1].ProjRf, output[1].ProjRa, output[1].ProjDiff, output[1].ProjW, output[1].ProjL)
                output[0] = TeamFull(output[0].Team, output[0].W, output[0].L, output[0].WPercent, '+' + str(output[0].W - output[2].W), output[0].Strk, output[0].Diff, output[0].Rf, output[0].RfG, output[0].Ra, output[0].RaG, output[0].PythW, output[0].PythL, 
                                    output[0].HomeWL, output[0].RoadWL, output[0].Over500WL, output[0].Under500WL, output[0].ProjRf, output[0].ProjRa, output[0].ProjDiff, output[0].ProjW, output[0].ProjL)
            elif (len(output) > 3):
                temp_GB += output[len(output) - 2].W - output[len(output) - 1].W
                output[len(output) - 1] = TeamFull(output[len(output) - 1].Team, output[len(output) - 1].W, output[len(output) - 1].L, output[len(output) - 1].WPercent, temp_GB, output[len(output) - 1].Strk, output[len(output) - 1].Diff, output[len(output) - 1].Rf, output[len(output) - 1].RfG, output[len(output) - 1].Ra, output[len(output) - 1].RaG, output[len(output) - 1].PythW, output[len(output) - 1].PythL, 
                                    output[len(output) - 1].HomeWL, output[len(output) - 1].RoadWL, output[len(output) - 1].Over500WL, output[len(output) - 1].Under500WL, output[len(output) - 1].ProjRf, output[len(output) - 1].ProjRa, output[len(output) - 1].ProjDiff, output[len(output) - 1].ProjW, output[len(output) - 1].ProjL)
        return output
    
    def _get_over_500_w(self, abbr: str) -> int:
        count = 0
        for i in range (len(self._team_schedule[abbr])):
            opp = self._team_schedule[abbr].iloc[i]['Opp']
            opp_w = int(self._team_schedule[opp].iloc[161]['W-L'].split('-')[0])
            if opp_w > 80 and 'W' in self._team_schedule[abbr].iloc[i]['W/L']:
                count += 1
        return count
    
    def _get_over_500_l(self, abbr: str) -> int:
        count = 0
        for i in range (len(self._team_schedule[abbr])):
            opp = self._team_schedule[abbr].iloc[i]['Opp']
            opp_w = int(self._team_schedule[opp].iloc[161]['W-L'].split('-')[0])
            if opp_w > 80 and 'L' in self._team_schedule[abbr].iloc[i]['W/L']:
                count += 1
        return count
    
    def _get_home_w(self, abbr: str) -> int:
        count = 0
        for i in range (len(self._team_schedule[abbr])):
            if self._team_schedule[abbr].iloc[i]['Home_Away'] == 'Home' and 'W' in self._team_schedule[abbr].iloc[i]['W/L']:
                count += 1
        return count
    
    def _get_home_l(self, abbr: str) -> int:
        count = 0
        for i in range (len(self._team_schedule[abbr])):
            if self._team_schedule[abbr].iloc[i]['Home_Away'] == 'Home' and 'L' in self._team_schedule[abbr].iloc[i]['W/L']:
                count += 1
        return count
            
    def _get_wins(self, team_num: int, teams: list[TeamFull]) -> int:
        if team_num >= 5:
            return -1
        return teams[team_num].W
    
    def _set_salaries(self):
        salaries = pd.read_csv("./ml_b/data/MLB-Salaries_2022.csv")
        for i in range(len(salaries)):
            split =  self._remove_accents(salaries.iloc[i]['Player']).replace('"', '').split(', ')
            salaries.at[i, 'Player'] = split[1] + ' ' + split[0]
        batting = self._player_batting[2022]
        payroll = []
        for i in range(len(batting)):
            name = batting.iloc[i]['Name']
            salary = 700000
            if name in list(salaries['Player']) and type(salaries[salaries['Player'] == name].iloc[0]['2022 ']) == str:
                salary = int(salaries[salaries['Player'] == name].iloc[0]['2022 '].replace(',', '').replace('"', '').replace(' ', '').replace('$', ''))
            payroll.append(salary)
        self._player_batting[2022]['Salary'] = payroll
        pitching = self._player_pitching[2022]
        payroll = []
        for i in range(len(pitching)):
            name = pitching.iloc[i]['Name']
            salary = 700000
            if name in list(salaries['Player']) and type(salaries[salaries['Player'] == name].iloc[0]['2022 ']) == str:
                salary = int(salaries[salaries['Player'] == name].iloc[0]['2022 '].replace(',', '').replace('"', '').replace(' ', '').replace('$', ''))
            payroll.append(salary)
        self._player_pitching[2022]['Salary'] = payroll
    
    def _set_extra_batter_stats(self, year: int) -> None:
        year_batting = self._player_batting[year]
        fielding_year = self._player_fielding[year]
        bwar = []
        awar = []
        tb = []
        pos = []
        player_id = []
        for i in range(len(year_batting)):
            name = year_batting.iloc[i]['Name']
            cur_pos = ''
            if len(fielding_year[fielding_year['Name'] == name]) > 0:
                cur_pos = fielding_year[fielding_year['Name'] == name].sort_values(by='Inn', ascending=False).iloc[0]['Pos']
            cur_tb = self._calc_tb(year_batting.iloc[i]['H'], year_batting.iloc[i]['2B'], year_batting.iloc[i]['3B'], year_batting.iloc[i]['HR'])
            cur_bwar = self._get_batter_bwar(year, name)
            cur_awar = round((year_batting.iloc[i]['WAR'] + cur_bwar) / 2, 2)
            bwar.append(cur_bwar)
            playerid = ''
            player_name = self._batting_bwar[self._batting_bwar['name_common'] == name]
            if len(player_name) > 0:
                playerid = player_name.iloc[0]['player_ID']
            player_id.append(playerid)
            awar.append(cur_awar)
            tb.append(cur_tb)
            pos.append(cur_pos)
        self._player_batting[year]['bWAR'] = bwar
        self._player_batting[year]['aWAR'] = awar
        self._player_batting[year]['TB'] = tb
        self._player_batting[year]['FLDPos'] = pos
        self._player_batting[year]['player_id'] = player_id
        
    def _set_extra_pitcher_stats(self, year: int) -> None:
        year_pitching = self._player_pitching[year]
        bwar = []
        awar = []
        pitches = []
        spin = []
        velo = []
        player_id = []
        for i in range(len(year_pitching)):
            name = year_pitching.iloc[i]['Name']
            cur_bwar = self._get_pitcher_bwar(year, name)
            cur_awar = round((year_pitching.iloc[i]['WAR'] + cur_bwar) / 2, 2)
            pitch = self._get_num_pitches(year, name)
            cur_spin = self._get_avr_spin_rate(year, name)
            max_velo = self._get_fastest_velo(year, name)
            playerid = ''
            player_name = self._pitching_bwar[self._pitching_bwar['name_common'] == name]
            if len(player_name) > 0:
                playerid = player_name.iloc[0]['player_ID']
            player_id.append(playerid)
            bwar.append(cur_bwar)
            awar.append(cur_awar)
            pitches.append(pitch)
            spin.append(cur_spin)
            velo.append(max_velo)
        self._player_pitching[year]['bWAR'] = bwar
        self._player_pitching[year]['aWAR'] = awar
        self._player_pitching[year]['NumPitches'] = pitches
        self._player_pitching[year]['AvrSpin'] = spin
        self._player_pitching[year]['MaxVelo'] = velo
        self._player_pitching[year]['player_id'] = player_id
        
    def _give_fg_names_pitch(self, year: int):
        year_statcast = self._statcast_pitcher[year]
        names = []
        for i in range(len(year_statcast)):
            name = self._remove_accents(year_statcast.iloc[i]['first_name'][1:] + ' ' + year_statcast.iloc[i]['last_name'])
            names.append(name)
        self._statcast_pitcher[year]['Name'] = names
        
    def _set_extra_fielder_stats(self, year: int) -> None:
        year_fielding = self._player_fielding[year]
        batting_year = self._player_batting[year]
        runs = []
        sb = []
        cs = []
        bsr = []
        for i in range(len(year_fielding)):
            name = year_fielding.iloc[i]['Name']
            cur_runs = 0
            cur_sb = 0
            cur_cs = 0
            cur_bsr = 0
            if len(batting_year[batting_year['Name'] == name]) > 0:
                cur_runs = batting_year[batting_year['Name'] == name].iloc[0]['R']
                cur_sb = batting_year[batting_year['Name'] == name].iloc[0]['SB']
                cur_cs = batting_year[batting_year['Name'] == name].iloc[0]['CS']
                cur_bsr = batting_year[batting_year['Name'] == name].iloc[0]['BsR']
            runs.append(cur_runs)
            sb.append(cur_sb)
            cs.append(cur_cs)
            bsr.append(cur_bsr)
        self._player_fielding[year]['R'] = runs
        self._player_fielding[year]['BRSB'] = sb
        self._player_fielding[year]['BRCS'] = cs
        self._player_fielding[year]['BsR'] = bsr
        
    def _get_num_pitches(self, year: int, name: str) -> int:
        yr = self._player_pitching[year]
        count = 0
        if not yr[yr['Name'] == name]['ff_avg_speed'].isnull().values.any():
            count += 1
        if not yr[yr['Name'] == name]['si_avg_speed'].isnull().values.any():
            count += 1
        if not yr[yr['Name'] == name]['fc_avg_speed'].isnull().values.any():
            count += 1
        if not yr[yr['Name'] == name]['sl_avg_speed'].isnull().values.any():
            count += 1
        if not yr[yr['Name'] == name]['ch_avg_speed'].isnull().values.any():
            count += 1
        if not yr[yr['Name'] == name]['cu_avg_speed'].isnull().values.any():
            count += 1
        if not yr[yr['Name'] == name]['fs_avg_speed'].isnull().values.any():
            count += 1
        if not yr[yr['Name'] == name]['kn_avg_speed'].isnull().values.any():
            count += 1
        return count
    
    def _get_avr_spin_rate(self, year: int, name: str) -> float:
        yr = self._player_pitching[year]
        tot_spin = 0.0
        count = 0
        if not yr[yr['Name'] == name]['ff_avg_spin'].isnull().values.any():
            tot_spin += yr[yr['Name'] == name].iloc[0]['ff_avg_spin']
            count += 1
        if not yr[yr['Name'] == name]['si_avg_spin'].isnull().values.any():
            tot_spin += yr[yr['Name'] == name].iloc[0]['si_avg_spin']
            count += 1
        if not yr[yr['Name'] == name]['fc_avg_spin'].isnull().values.any():
            tot_spin += yr[yr['Name'] == name].iloc[0]['fc_avg_spin']
            count += 1
        if not yr[yr['Name'] == name]['sl_avg_spin'].isnull().values.any():
            tot_spin += yr[yr['Name'] == name].iloc[0]['sl_avg_spin']
            count += 1
        if not yr[yr['Name'] == name]['ch_avg_spin'].isnull().values.any():
            tot_spin += yr[yr['Name'] == name].iloc[0]['ch_avg_spin']
            count += 1
        if not yr[yr['Name'] == name]['cu_avg_spin'].isnull().values.any():
            tot_spin += yr[yr['Name'] == name].iloc[0]['cu_avg_spin']
            count += 1
        if not yr[yr['Name'] == name]['fs_avg_spin'].isnull().values.any():
            tot_spin += yr[yr['Name'] == name].iloc[0]['fs_avg_spin']
            count += 1
        if not yr[yr['Name'] == name]['kn_avg_spin'].isnull().values.any():
            tot_spin += yr[yr['Name'] == name].iloc[0]['kn_avg_spin']
            count += 1
        if count == 0:
            return 0.0
        return tot_spin / count
        
    def _get_fastest_velo(self, year: int, name: str) -> float:
        yr = self._player_pitching[year]
        max = 0
        if not yr[yr['Name'] == name]['ff_avg_speed'].isnull().values.any() and yr[yr['Name'] == name].iloc[0]['ff_avg_speed'] > max:
            max = yr[yr['Name'] == name].iloc[0]['ff_avg_speed']
        if not yr[yr['Name'] == name]['si_avg_speed'].isnull().values.any() and yr[yr['Name'] == name].iloc[0]['si_avg_speed'] > max:
            max = yr[yr['Name'] == name].iloc[0]['si_avg_speed']
        if not yr[yr['Name'] == name]['fc_avg_speed'].isnull().values.any() and yr[yr['Name'] == name].iloc[0]['fc_avg_speed'] > max:
            max = yr[yr['Name'] == name].iloc[0]['fc_avg_speed']
        if not yr[yr['Name'] == name]['sl_avg_speed'].isnull().values.any() and yr[yr['Name'] == name].iloc[0]['sl_avg_speed'] > max:
            max = yr[yr['Name'] == name].iloc[0]['sl_avg_speed']
        if not yr[yr['Name'] == name]['ch_avg_speed'].isnull().values.any() and yr[yr['Name'] == name].iloc[0]['ch_avg_speed'] > max:
            max = yr[yr['Name'] == name].iloc[0]['ch_avg_speed']
        if not yr[yr['Name'] == name]['cu_avg_speed'].isnull().values.any() and yr[yr['Name'] == name].iloc[0]['cu_avg_speed'] > max:
            max = yr[yr['Name'] == name].iloc[0]['cu_avg_speed']
        if not yr[yr['Name'] == name]['fs_avg_speed'].isnull().values.any() and yr[yr['Name'] == name].iloc[0]['fs_avg_speed'] > max:
            max = yr[yr['Name'] == name].iloc[0]['fs_avg_speed']
        if not yr[yr['Name'] == name]['kn_avg_speed'].isnull().values.any() and yr[yr['Name'] == name].iloc[0]['kn_avg_speed'] > max:
            max = yr[yr['Name'] == name].iloc[0]['kn_avg_speed']
        return max
    
    def _remove_all_accents(self):
        for i in range(2022, 2014, -1):
            for j in range(len(self._statcast_fielder[i])):
                self._statcast_fielder[i].at[j, 'first_name'] = self._remove_accents(self._statcast_fielder[i].iloc[j]['first_name'])[1:]
                self._statcast_fielder[i].at[j, 'last_name'] = self._remove_accents(self._statcast_fielder[i].iloc[j]['last_name'])
        for i in range(len(self._batting_bwar)):
            self._batting_bwar.at[i, 'name_common'] = self._remove_accents(self._batting_bwar.iloc[i]['name_common'])
        for i in range(len(self._pitching_bwar)):
            self._pitching_bwar.at[i, 'name_common'] = self._remove_accents(self._pitching_bwar.iloc[i]['name_common'])
        
    def _remove_accents(self, name: str) -> str:
        temp = name.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n').replace('ü', 'u').replace('Á', 'A')
        return temp.replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').replace('Ú', 'U').replace('Ñ', 'N').replace('Ü', 'U')
    
    def _get_offense_stat(self, year: int, stat: str, player: str):
        player_stats = self._player_batting[year][self._player_batting[year]['Name'] == player]
        return player_stats.iloc[0][stat]
                
    def _get_batter_bwar(self, year: int, player: str) -> float:
        bwar_year = self._batting_bwar[self._batting_bwar['year_ID'] == year]
        player_bwar = bwar_year[bwar_year['name_common'] == player]
        bwar_sum = 0.0
        for x in range(len(player_bwar)):
            bwar_sum += player_bwar.iloc[x]['WAR']
        return bwar_sum
    
    def _get_pitcher_bwar(self, year: int, player: str) -> float:
        bwar_year = self._pitching_bwar[self._pitching_bwar['year_ID'] == year]
        player_bwar = bwar_year[bwar_year['name_common'] == player]
        bwar_sum = 0.0
        for x in range(len(player_bwar)):
            bwar_sum += player_bwar.iloc[x]['WAR']
        return bwar_sum
        
    def _calc_pyth_wins(self, runsF: int, runsA: int) -> int:
        return int((int(runsF) ** 1.83) / ((int(runsF) ** 1.83) + (int(runsA) ** 1.83)) * 162)
        
    def _get_streak(self, team: str) -> str:
        counter = self._team_schedule[team_abbr[team]].iloc[161]['Streak']
        game = ''
        if counter > 0:
            game = 'W'
        else:
            game = 'L'
        return game + str(abs(counter))
    
    def _calc_tb(self, hits: int, doubles: int, triples: int, homers: int) -> int:
        return (hits - doubles - triples - homers) + (2 * doubles) + (3 * triples) + (4 * homers)