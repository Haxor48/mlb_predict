from pybaseball import batting_stats, pitching_stats, team_batting, fielding_stats, statcast_outfielder_jump, statcast_sprint_speed, statcast_outs_above_average, statcast_batter_expected_stats, \
    statcast_pitcher_exitvelo_barrels, statcast_batter_exitvelo_barrels, statcast_pitcher_expected_stats, playerid_lookup, statcast_batter, standings, batting_stats_bref, schedule_and_record, \
    statcast_pitcher_pitch_arsenal, bwar_bat, cache, batting_stats_bref, player_search_list
from pybaseball.lahman import salaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from bs4 import BeautifulSoup
import requests
from ml_b import get_data


if __name__ == '__main__':
    data = get_data.Data()
    data.get_data('', True, True, True, False, False, False, False, False)
    print(data.get_player_top_percentiles('Aaron Judge', True))
    