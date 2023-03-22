from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('leaders/', views.leaders, name='leaders'),
    path('leaders/hitting/', views.hitting_leaders, name='hitting-leaders'),
    path('leaders/pitching/', views.pitching_leaders, name='pitching-leaders'),
    path('leaders/fielding/', views.fielding_leaders, name='fielding-leaders'),
    path('teams/', views.teams, name='teams'),
    path('teams/team/', views.team, name='team'),
    path('standings/', views.standings, name='standings'),
    path('salaries/', views.salaries, name='salaries'),
    path('players/', views.players, name='players'),
    path('players/player/', views.player, name='player')
]