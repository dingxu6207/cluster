# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 11:38:33 2021

@author: dingxu
"""
#!pip install astroquery

#import module
from astroquery.simbad import Simbad

#get a simbad instance
simbad = Simbad()

#add spectral type parameters for VOTable request
simbad.add_votable_fields('sptype')

#request
result_table = simbad.query_object("gam cas")
result_table.pprint(show_unit=True)

import ipyaladin.aladin_widget as ipyal
aladin = ipyal.Aladin(target='gam cas', fov = 1)

aladin.add_table(result_table)

#record render in jpeg
aladin.get_JPEG_thumbnail()