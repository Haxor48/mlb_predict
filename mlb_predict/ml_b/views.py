from django.http import HttpResponse
from django.template import loader
from . import get_data

data = get_data.Data()

def _get_params(response, mode: int) -> list[tuple]:
    url = response.get_full_path()
    output = [(), (), (), ()]
    if mode == 1:
        if 'hit' not in url:
            output[0] = ('aWAR', 0, 10, False, False)
            output[1] = ('aWAR', 0, 10, False, False)
            output[2] = ('Def', 0, 10, False, False)
        else:
            split = url.split('?')
            hit = split[1].split('=')[1].split(',')
            pitch = split[2].split('=')[1].split(',')
            field = split[3].split('=')[1].split(',')
            output[0] = (hit[0], int(hit[1]), int(hit[2]), _str_to_bool(hit[3]), _str_to_bool(hit[4]))
            output[1] = (pitch[0], int(pitch[1]), int(pitch[2]), _str_to_bool(pitch[3]), _str_to_bool(pitch[4]))
            output[2] = (field[0], int(field[1]), int(field[2]), _str_to_bool(field[3]), _str_to_bool(field[4]))
    elif mode == 2:
        if 'basic' not in url:
            if 'hitting' in url:
                output[0] = ('TB', 0, 10, False, False)
                output[1] = ('aWAR', 0, 10, False, False)
                output[2] = ('xwOBA', 0, 10, False, False)
                output[3] = ('aWAR', 0, 10, False, False)
            else:
                output[0] = ('W', 0, 10, False, False)
                output[1] = ('aWAR', 0, 10, False, False)
                output[2] = ('AvrSpin', 0, 10, False, False)
                output[3] = ('aWAR', 0, 10, False, False)
        else:
            split = url.split('?')
            basic = split[1].split('=')[1].split(',')
            advanced = split[2].split('=')[1].split(',')
            statcast = split[3].split('=')[1].split(',')
            proj = split[4].split('=')[1].split(',')
            output[0] = (basic[0], int(basic[1]), int(basic[2]), _str_to_bool(basic[3]), _str_to_bool(basic[4]))
            output[1] = (advanced[0], int(advanced[1]), int(advanced[2]), _str_to_bool(advanced[3]), _str_to_bool(advanced[4]))
            output[2] = (statcast[0], int(statcast[1]), int(statcast[2]), _str_to_bool(statcast[3]), _str_to_bool(statcast[4]))
            output[3] = (proj[0], int(proj[1]), int(proj[2]), _str_to_bool(proj[3]), _str_to_bool(proj[4]))
    else:
        if 'basic' not in url:
            output[0] = ('Def', 0, 10, False, False)
            output[1] = ('outs_above_average_x', 0, 10, False, False)
        else:
            split = url.split('?')
            basic = split[1].split('=')[1].split(',')
            statcast = split[2].split('=')[1].split(',')
            output[0] = (basic[0], int(basic[1]), int(basic[2]), _str_to_bool(basic[3]), _str_to_bool(basic[4]))
            output[1] = (statcast[0], int(statcast[1]), int(statcast[2]), _str_to_bool(statcast[3]), _str_to_bool(statcast[4]))
    return output

def home(response):
    template = loader.get_template("index.html")
    data.get_data(response.build_absolute_uri, True, True, True, True, True, True, True, False)
    context = {
        'hitting_leaders': data.get_home_hitting_leaders(),
        'pitching_leaders': data.get_home_pitching_leaders(),
        'al_east': data.get_standings(0),
        'al_central': data.get_standings(1),
        'al_west': data.get_standings(2),
        'nl_east': data.get_standings(3),
        'nl_central': data.get_standings(4),
        'nl_west': data.get_standings(5),
        'url': data.get_url()
    }
    return HttpResponse(template.render(context, response))

def leaders(response):
    template = loader.get_template("leaders.html")
    data.get_data(response.build_absolute_uri, True, True, True, False, False, False, False, False)
    leaders = _get_params(response, 1)
    context = {
        'hitting_leaders': data.get_standard_hitting_leaders(leaders[0][0], leaders[0][1], leaders[0][2], leaders[0][3]),
        'pitching_leaders': data.get_standard_pitching_leaders(leaders[1][0], leaders[1][1], leaders[1][2], leaders[1][3]),
        'fielding_leaders': data.get_standard_fielding_leaders(*leaders[2])
    }
    return HttpResponse(template.render(context, response))

def hitting_leaders(response):
    template = loader.get_template("hitting-leaders.html")
    data.get_data(response.build_absolute_uri, True, False, True, False, False, False, False, False)
    leaders = _get_params(response, 2)
    context = {
        'basic_batting': data.get_basic_batter_leaders(*leaders[0]),
        'advanced_batting': data.get_advanced_batter_leaders(*leaders[1]),
        'statcast_batting': data.get_statcast_batter_leaders(*leaders[2]),
        'projected_batting': data.get_projected_hitting_leaders(*leaders[3])
    }
    return HttpResponse(template.render(context, response))

def pitching_leaders(response):
    template = loader.get_template("pitching-leaders.html")
    data.get_data(response.build_absolute_uri, False, True, False, False, False, False, False, False)
    leaders = _get_params(response, 2)
    context = {
        'basic_pitching': data.get_basic_pitcher_leaders(*leaders[0]),
        'advanced_pitching': data.get_advanced_pitcher_leaders(*leaders[1]),
        'statcast_pitching': data.get_statcast_pitcher_leaders(*leaders[2]),
        'projected_pitching': data.get_projected_pitching_leaders(*leaders[3])
    }
    return HttpResponse(template.render(context, response))

def fielding_leaders(response):
    template = loader.get_template("fielding-leaders.html")
    data.get_data(response.build_absolute_uri, False, False, True, False, False, False, False, False)
    leaders = _get_params(response, 3)
    context = {
        'fielding_leaders_basic': data.get_standard_fielding_leaders(*leaders[0]),
        'fielding_leaders_statcast': data.get_statcast_fielder_leaders(*leaders[1])
    }
    return HttpResponse(template.render(context, response))

def teams(response):
    template = loader.get_template("teams.html")
    data.get_data(response.build_absolute_uri, False, False, False, False, False, False, False, True)
    context = {
        'img': data.get_team_imgs()
    }
    return HttpResponse(template.render(context, response))

def team(response):
    template = loader.get_template("team.html")
    team = response.build_absolute_uri().split('?')[1]
    data.get_data(response.build_absolute_uri(), True, True, True, True, True, True, False, True)
    context = {
        'team_hitting': data.get_team_hitting(team),
        'team_pitching': data.get_team_pitching(team),
        'hitting_leaders': data.get_team_hitting_leaders(team),
        'pitching_leaders': data.get_team_pitching_leaders(team),
        'fielding_leaders': data.get_team_fielding_leaders(team),
        'info': data.get_team_info(team)
    }
    return HttpResponse(template.render(context, response))

def standings(response):
    template = loader.get_template("standings.html")
    data.get_data(response.build_absolute_uri(), False, False, False, True, True, True, True, False)
    context = {
        'al_east': data.get_full_standings(0),
        'al_central': data.get_full_standings(1),
        'al_west': data.get_full_standings(2),
        'nl_east': data.get_full_standings(3),
        'nl_central': data.get_full_standings(4),
        'nl_west': data.get_full_standings(5),
    }
    context['al_wildcard'] =  data.get_wildcard(context['al_west'], context['al_central'], context['al_east'])
    context['nl_wildcard'] = data.get_wildcard(context['nl_west'], context['nl_central'], context['nl_east'])
    return HttpResponse(template.render(context, response))

def salaries(response):
    template = loader.get_template("salary-projector.html")
    data.get_data(response.build_absolute_uri(), True, True, False, False, False, False, False, False)
    context = {
        'players': data.get_player_salaries()
    }
    return HttpResponse(template.render(context, response))

def players(response):
    template = loader.get_template("players.html")
    data.get_data(response.build_absolute_uri(), True, True, True, False, False, False, False, False)
    context = {
        'players': data.get_players()
    }
    return HttpResponse(template.render(context, response))

def player(response):
    template = loader.get_template("player.html")
    data.get_data(response.build_absolute_uri(), True, True, True, False, False, False, False, False)
    name = ''
    split = []
    if '$' in response.build_absolute_uri():
        split = response.build_absolute_uri().split('?')[1].split('$')
        name = split[0].replace('+', ' ')
        split = split[1].split(',')
    else:
        name = response.build_absolute_uri().split('?')[1].replace('+', ' ')
    is_hitter = data.is_hitter(name)
    if len(split) < 1:
        if is_hitter:
            split = ['f'] * 6
        else:
            split = ['f'] * 4
    percentiles = data.get_player_top_percentiles(name, is_hitter)
    context = {
        'name': name,
        'isHitter':  is_hitter,
        'info': data.get_player_info(name),
        'top_percentiles': percentiles[0],
        'bottom_percentiles': percentiles[1]
    }
    if is_hitter:
        context['basic_batting'] = data.get_player_batter_basic(name, _str_to_bool(split[0]))
        context['advanced_batting'] = data.get_player_batter_advanced(name, _str_to_bool(split[1]))
        context['statcast_batting'] = data.get_player_batter_statcast(name, _str_to_bool(split[2]))
        context['fielding_basic'] = data.get_player_fielding_standard(name, _str_to_bool(split[3]))
        context['fielding_statcast'] = data.get_player_fielder_statcast(name, _str_to_bool(split[4]))
        context['projected_batting'] = data.get_player_projected_hitting(name, _str_to_bool(split[5]))
    else:
        context['basic_pitching'] = data.get_player_pitcher_basic(name, _str_to_bool(split[0]))
        context['advanced_pitching'] = data.get_player_pitcher_advanced(name, _str_to_bool(split[1]))
        context['statcast_pitching'] = data.get_player_pitcher_statcast(name, _str_to_bool(split[2]))
        context['projected_pitching'] = data.get_player_projected_pitching(name, _str_to_bool(split[3]))
    return HttpResponse(template.render(context, response))

def _str_to_bool(text: str) -> bool:
    return 't' in text

