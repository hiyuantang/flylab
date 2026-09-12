"""Versioned, millimetre-scale worlds shared by physics, vision and rendering.

Procedural original assets. Every listed primitive is a static solid in MuJoCo;
the renderer consumes the same dimensions and quaternion. Rooms are cutaways
with two enclosing walls and no ceiling, not miniatures of human furniture.
"""
from functools import lru_cache
import hashlib
import json
import math

import numpy as np

SCENE_IDS = ('lab', 'kitchen', 'living-room', 'bedroom', 'garden')


class Builder:
    def __init__(self, key, title, description, extent, spawn, location, floor, background, odor, sound):
        self.scene = dict(id=key, title=title, description=description, extent=extent, spawn=spawn,
                          spawn_label=location, floor_color=floor, background=background,
                          odor_source=odor, sound_source=sound, objects=[],
                          camera={'target': [0, 100, 700], 'position': [4400, -4900, 3900]})

    def geom(self, label, shape, pos, size, color, *, quat=(1, 0, 0, 0), material='matte', category='Furniture'):
        obj = dict(id=f"world_{len(self.scene['objects']):03d}", label=label, shape=shape,
                   position=list(pos), size=list(size), quaternion=list(quat), color=color,
                   material=material, category=category)
        self.scene['objects'].append(obj)

    def box(self, label, pos, dimensions, color, **kw):
        self.geom(label, 'box', pos, [x / 2 for x in dimensions], color, **kw)

    def cylinder(self, label, pos, radius, height, color, **kw):
        self.geom(label, 'cylinder', pos, [radius, height / 2], color, **kw)

    def ellipsoid(self, label, pos, radii, color, **kw):
        self.geom(label, 'ellipsoid', pos, radii, color, **kw)

    def rod(self, label, start, end, radius, color, **kw):
        a, b = np.array(start), np.array(end)
        direction = (b - a) / np.linalg.norm(b - a)
        q = np.array([1 + direction[2], -direction[1], direction[0], 0.])
        q = q / np.linalg.norm(q) if np.linalg.norm(q) > 1e-8 else np.array([0., 1., 0., 0.])
        self.cylinder(label, (a + b) / 2, radius, np.linalg.norm(b - a), color, quat=q.tolist(), **kw)

    def room(self, width, depth, wall='#e5dfd2'):
        for label, pos, dims in [('Back wall', (0, depth / 2, 1250), (width, 70, 2500)),
                                  ('Side wall', (-width / 2, 0, 1250), (70, depth, 2500))]:
            self.box(label, pos, dims, wall, category='Architecture')
        self.box('Back skirting', (0, depth / 2 - 40, 55), (width, 18, 110), '#f3ecdf', category='Architecture')
        self.box('Side skirting', (-width / 2 + 40, 0, 55), (18, depth, 110), '#f3ecdf', category='Architecture')
        # Flush frame and bright window panels: opaque optical proxies, not glass refraction.
        self.box('Window frame', (-width / 2 + 45, 350, 1550), (35, 1280, 1250), '#fcf7e8', category='Architecture')
        self.box('Window daylight', (-width / 2 + 65, 350, 1550), (8, 1160, 1130), '#aecdd1', material='luminous', category='Architecture')
        self.box('Window vertical mullion', (-width / 2 + 75, 350, 1550), (20, 26, 1140), '#f8f0e1', category='Architecture')
        self.box('Window horizontal mullion', (-width / 2 + 75, 350, 1550), (20, 1170, 26), '#f8f0e1', category='Architecture')
        self.box('Window sill', (-width / 2 + 110, 350, 920), (180, 1380, 45), '#e3d4b9', material='wood', category='Architecture')

    def table(self, label, x, y, top, width, depth, color='#ae7b4e'):
        self.box(f'{label} top', (x, y, top - 20), (width, depth, 40), color, material='wood')
        self.box(f'{label} apron', (x, y, top - 70), (width - 65, depth - 65, 60), color, material='wood')
        for dx in (-1, 1):
            for dy in (-1, 1):
                self.box(f'{label} leg', (x + dx * (width / 2 - 45), y + dy * (depth / 2 - 45), (top - 100) / 2),
                         (45, 45, top - 100), '#725035', material='wood')

    def chair(self, x, y, facing=1):
        self.table('Chair', x, y, 450, 420, 400, '#c28c56')
        self.box('Chair back', (x, y - 180 * facing, 680), (420, 35, 410), '#c28c56', material='wood')
        self.box('Chair seat pad', (x, y, 470), (370, 340, 35), '#d8d1be', material='fabric')

    def plant(self, x, y, base, scale=1.):
        self.cylinder('Terracotta planter', (x, y, base + 130 * scale), 125 * scale, 260 * scale, '#b97550', category='Plant')
        self.cylinder('Planter rim', (x, y, base + 250 * scale), 140 * scale, 35 * scale, '#cc8b62', category='Plant')
        self.cylinder('Potting soil', (x, y, base + 271 * scale), 118 * scale, 8 * scale, '#453c2c', category='Plant')
        for i in range(9):
            a = i * 2.4
            end = [x + math.cos(a) * 170 * scale, y + math.sin(a) * 170 * scale, base + (450 + i % 3 * 80) * scale]
            self.rod('Plant stem', [x, y, base + 270 * scale], end, 4 * scale, '#426342', category='Plant')
            q = [math.cos(a / 2), 0, 0, math.sin(a / 2)]
            self.ellipsoid('Leaf', end, [100 * scale, 40 * scale, 7 * scale], ['#4e7950', '#608a52', '#386347'][i % 3], quat=q, category='Plant')

    def book(self, x, y, z, width, depth, height, color, label='Book'):
        self.box(f'{label} pages', (x, y, z + height / 2), (width - 4, depth - 8, height - 4), '#e9dfc8')
        for level in (z + 1, z + height - 1):
            self.box(f'{label} cover', (x, y, level), (width, depth, 2), color)
        self.box(f'{label} spine', (x - width / 2 + 2, y, z + height / 2), (4, depth, height), color)

    def mug(self, x, y, z, color='#e4d3ac'):
        self.cylinder('Mug base', (x, y, z + 3), 37, 6, color)
        for i in range(20):
            a = i * math.tau / 20
            self.box('Mug wall', (x + 35 * math.cos(a), y + 35 * math.sin(a), z + 43), (6, 12, 80), color,
                     quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)])
        self.cylinder('Coffee surface (static)', (x, y, z + 66), 31, 2, '#38291c')
        for a, b in [((x + 35, y, z + 65), (x + 57, y, z + 65)), ((x + 57, y, z + 65), (x + 57, y, z + 20)), ((x + 57, y, z + 20), (x + 35, y, z + 20))]:
            self.rod('Mug handle', a, b, 5, color)

    def crumbs(self, x, y, z):
        for i, (dx, dy, r) in enumerate([(0, 0, .7), (3.5, 2, .5), (6, -1, .9), (-3, 5, .4), (9, 5, 1.2), (14, -4, .6)]):
            self.ellipsoid('Food crumb', (x + dx, y + dy, z + r * .55), (r, r * .7, r * .55), '#b17b39', category='Food')

    def finish(self):
        raw = json.dumps(self.scene, sort_keys=True, separators=(',', ':'))
        self.scene['version'] = hashlib.sha256(raw.encode()).hexdigest()
        return self.scene


def kitchen():
    b = Builder('kitchen', 'Kitchen', 'Sage cabinetry, breakfast table and a sunlit counter with coffee, fruit and crumbs.',
                [3600, 3000, 2500], [250, 1050, 900], 'Kitchen counter', '#bca17c', '#d8dfda', [258, 1052, 900.5], [240, 1040, 901.5])
    b.room(3600, 3000, '#e9e4d5')
    b.box('Worktop', (0, 1160, 875), (3120, 620, 50), '#eee6d8', material='stone')
    for x in (-1250, -625, 0, 625, 1250):
        b.box('Base cabinet', (x, 1190, 410), (600, 520, 820), '#788e7a')
        b.box('Cabinet door', (x, 920, 430), (570, 22, 740), '#90a18a')
        b.box('Brass pull', (x, 900, 650), (180, 12, 12), '#a38648', material='metal')
        b.box('Upper cabinet', (x, 1310, 1860), (580, 300, 700), '#d8d7c4')
        b.box('Upper door inset', (x, 1150, 1860), (520, 15, 630), '#e8e6d6')
        b.box('Upper cabinet pull', (x + 190, 1135, 1600), (10, 10, 100), '#a38648', material='metal')
    b.box('Backsplash', (0, 1450, 1190), (3100, 12, 570), '#d0dad4', material='tile')
    for x in range(-1500, 1501, 250):
        b.box('Tile grout', (x, 1442, 1190), (3, 2, 560), '#9caea7', category='Architecture')
    for z in (950, 1100, 1250, 1400):
        b.box('Tile grout', (0, 1442, z), (3090, 2, 3), '#9caea7', category='Architecture')
    b.box('Cooktop', (-950, 1140, 906), (500, 470, 12), '#283437', material='metal')
    for x in (-1080, -820):
        for y in (1020, 1260):
            b.cylinder('Stove ring', (x, y, 916), 80, 9, '#131d20', material='metal')
    b.cylinder('Saucepan', (-1080, 1260, 985), 90, 130, '#b6bdb8', material='metal')
    b.rod('Pan handle', (-1000, 1260, 1010), (-800, 1260, 1010), 15, '#2e3431')
    b.box('Sink basin', (950, 1160, 905), (460, 390, 10), '#707f81', material='metal')
    b.box('Sink inset', (950, 1160, 912), (400, 330, 5), '#394e55', material='metal')
    b.rod('Faucet upright', (950, 1390, 910), (950, 1390, 1200), 16, '#b8c2bf', material='metal')
    b.rod('Faucet spout', (950, 1390, 1200), (950, 1220, 1200), 16, '#b8c2bf', material='metal')
    b.box('Refrigerator', (-1450, -850, 950), (650, 650, 1900), '#d3d9d6', material='metal')
    b.box('Fridge door', (-1450, -1190, 1060), (620, 30, 1660), '#e3e6df', material='metal')
    b.box('Fridge handle', (-1700, -1220, 1000), (22, 20, 580), '#66716d', material='metal')
    b.table('Breakfast table', -100, -650, 750, 1300, 750)
    b.chair(-450, -1300); b.chair(250, -1300)
    b.cylinder('Fruit plate', (-150, -650, 758), 125, 12, '#ddd3b5')
    for x, y in [(-170, -620), (-80, -665), (-205, -715)]:
        b.ellipsoid('Orange', (x, y, 800), [40, 39, 41], '#cc8037', category='Food')
    b.mug(350, 1140, 900, '#bbc8af'); b.crumbs(258, 1052, 900)
    b.box('Cutting board', (-250, 1100, 910), (320, 220, 20), '#bd925b', material='wood')
    b.ellipsoid('Sliced bread', (-250, 1100, 937), [110, 75, 18], '#d6b87a', category='Food')
    b.plant(-1350, 1160, 900, .45)
    return b.finish()


def living_room():
    b = Builder('living-room', 'Living room', 'A warm reading room with linen seating, walnut furniture, books and plants.',
                [4200, 3600, 2500], [80, -80, 420], 'Coffee table', '#ad8961', '#d6dbd7', [88, -75, 420.5], [100, -95, 421.5])
    b.room(4200, 3600, '#dfdfcf')
    b.box('Woven rug', (0, 100, 4), (2700, 2100, 8), '#c4baa1', material='fabric')
    for x in (-1290, 1290):
        b.box('Rug border', (x, 100, 8.2), (35, 2050, .4), '#9e8060', material='fabric')
    b.box('Sofa plinth', (0, 1130, 190), (2100, 850, 220), '#6b5644', material='wood')
    b.box('Sofa back', (0, 1480, 580), (2100, 190, 650), '#9aab9e', material='fabric')
    for x in (-1020, 1020):
        b.box('Sofa arm', (x, 1110, 440), (200, 850, 450), '#aab6a6', material='fabric')
    for x in (-620, 0, 620):
        b.box('Seat cushion', (x, 1080, 385), (590, 650, 170), '#bbc3ad', material='fabric')
        b.ellipsoid('Back cushion', (x, 1350, 640), (290, 95, 245), '#b5bfaa', material='fabric')
    b.ellipsoid('Ochre pillow', (-730, 1180, 620), [160, 100, 180], '#bb914e', material='fabric')
    b.ellipsoid('Terracotta pillow', (740, 1180, 620), [160, 100, 180], '#b37559', material='fabric')
    for x in (-650, 0, 650):
        b.box('Picture frame', (x, 1745, 1670), (420, 28, 520), '#8c7351', material='wood')
        b.box('Picture paper', (x, 1728, 1670), (376, 4, 476), '#ece3cd')
        b.rod('Botanical print stem', (x, 1724, 1490), (x + 20, 1724, 1840), 4, '#758566')
        for dx, z in [(-45, 1590), (50, 1690), (-40, 1760)]:
            b.ellipsoid('Botanical print leaf', (x + dx, 1722, z), (55, 2, 24), '#95a183')
    b.table('Coffee table', 0, 0, 420, 1100, 620, '#845c3c')
    b.book(-230, 80, 420, 210, 280, 30, '#436a68', 'Botany book')
    b.book(-210, 80, 450, 190, 240, 24, '#b87e54', 'Notebook')
    b.mug(290, 90, 420); b.crumbs(88, -75, 420)
    b.box('Media console', (1350, -1000, 260), (1050, 400, 520), '#876a4a', material='wood')
    for x in (1020, 1350, 1680):
        b.box('Console door', (x, -1210, 265), (300, 20, 425), '#ad8859', material='wood')
    b.box('Television stand', (1350, -990, 548), (420, 160, 20), '#273238', material='metal')
    b.box('Television', (1350, -960, 910), (1120, 48, 650), '#223239')
    b.box('Screen reflection', (1350, -931, 910), (1060, 3, 590), '#344e56')
    b.box('Television rear panel', (1350, -989, 900), (520, 8, 270), '#2c3436')
    b.box('Bookcase frame', (-1500, 1350, 1000), (600, 340, 2000), '#6d533c', material='wood')
    for z in (300, 720, 1140, 1560):
        b.box('Open shelf recess', (-1500, 1169, z + 160), (530, 5, 330), '#332f28')
        b.box('Shelf board', (-1500, 1130, z), (570, 400, 28), '#a5855d', material='wood')
        for i in range(6):
            b.box('Book spine', (-1710 + i * 75, 1090, z + 150), (48 + i % 2 * 12, 170, 245 + i % 3 * 20), ['#6b8684', '#aa7959', '#b9ad84'][i % 3])
    b.plant(-1550, -750, 0, 1.5)
    b.cylinder('Floor lamp base', (1500, 1350, 20), 180, 40, '#65584a', material='metal')
    b.rod('Lamp stem', (1500, 1350, 40), (1500, 1350, 1650), 16, '#aa8e54', material='metal')
    b.cylinder('Linen lamp shade', (1500, 1350, 1660), 230, 350, '#eadbb1', material='luminous')
    return b.finish()


def bedroom():
    b = Builder('bedroom', 'Bedroom', 'Soft bedding, oak storage and a bedside reading surface in morning light.',
                [4000, 3500, 2500], [650, 850, 520], 'Bedside table', '#b39477', '#d9d8dd', [658, 852, 520.5], [640, 840, 521.5])
    b.room(4000, 3500, '#dfd9d8')
    b.box('Bedside rug', (-400, 200, 5), (2500, 2500, 10), '#b9b7ae', material='fabric')
    b.box('Bed frame', (-450, 400, 230), (1620, 2120, 380), '#8f6c50', material='wood')
    b.box('Headboard', (-450, 1500, 780), (1730, 90, 1150), '#a48c7d', material='fabric')
    b.box('Mattress', (-450, 400, 500), (1550, 2050, 220), '#ded8c4', material='fabric')
    b.box('Duvet', (-450, 80, 635), (1590, 1440, 110), '#abb8b4', material='fabric')
    b.box('Duvet folded edge', (-450, 730, 682), (1590, 180, 70), '#c4cfca', material='fabric')
    b.box('Woven throw', (-450, -420, 705), (1600, 410, 35), '#b98761', material='fabric')
    for x in (-850, -50):
        b.ellipsoid('Pillow', (x, 1130, 670), [340, 210, 80], '#f0e7d7', material='fabric')
    b.table('Bedside table', 650, 900, 520, 500, 430, '#bb956c')
    b.box('Bedside drawer', (650, 900, 370), (425, 360, 190), '#be9c79', material='wood')
    b.cylinder('Drawer knob', (650, 709, 370), 17, 24, '#9d804d', quat=[.70710678, .70710678, 0, 0], material='metal')
    b.book(530, 1000, 520, 160, 210, 25, '#73858c', 'Bedside book')
    b.crumbs(658, 852, 520)
    b.cylinder('Bedside lamp foot', (800, 1000, 533), 60, 26, '#837151', material='metal')
    b.rod('Bedside lamp stem', (800, 1000, 540), (800, 1000, 860), 9, '#9f8964', material='metal')
    b.cylinder('Bedside shade', (800, 1000, 870), 105, 200, '#eddfbc', material='luminous')
    b.box('Wardrobe', (1410, 1050, 1080), (900, 580, 2160), '#a68260', material='wood')
    for x in (1185, 1635):
        b.box('Wardrobe door', (x, 749, 1080), (432, 24, 2090), '#c0a17c', material='wood')
        b.box('Wardrobe handle', (x + (150 if x < 1400 else -150), 728, 1040), (14, 14, 260), '#655746', material='metal')
    b.table('Writing desk', -1150, -1200, 750, 1250, 550, '#b89a72')
    b.chair(-1150, -650, facing=-1)
    b.book(-1300, -1200, 750, 240, 300, 20, '#667f83', 'Sketchbook')
    b.cylinder('Pencil cup', (-800, -1200, 810), 35, 120, '#a6b2ac')
    for i in range(5):
        b.rod('Pencil', (-820 + i * 9, -1200, 780), (-820 + i * 12, -1200, 940), 3, '#b78142')
    b.box('Picture frame', (600, 1700, 1700), (550, 30, 700), '#7d674c', material='wood')
    b.box('Picture mat', (600, 1681, 1700), (490, 8, 640), '#f2e6cd')
    b.box('Botanical print', (600, 1674, 1700), (360, 4, 480), '#879b87')
    b.plant(-1700, -250, 0, 1.2)
    return b.finish()


def garden():
    b = Builder('garden', 'Garden', 'A walled herb garden with paving stones, raised beds, pots and a timber bench.',
                [5000, 4000, 1900], [0, -100, 15], 'Central paving stone', '#74684d', '#cadaca', [8, -95, 15.5], [-10, -110, 16.5])
    b.scene['camera'] = {'target': [0, 100, 420], 'position': [5200, -5700, 4500]}
    b.box('Garden wall', (0, 1950, 650), (5000, 90, 1300), '#c9b294', category='Architecture')
    for z in (250, 500, 750, 1000, 1250):
        b.box('Wall mortar', (0, 1902, z), (4970, 3, 8), '#9b917b', category='Architecture')
    for x in (-1920, -1440, -960, -480, 0, 480, 960, 1440, 1920):
        b.box('Paving stone', (x, -100, 7.5), (440, 490, 15), ['#a7a69a', '#b7b5a4', '#929c94'][abs(x // 480) % 3], material='stone', category='Terrain')
    for x in (-1300, 100, 1450):
        b.box('Raised bed soil', (x, 1150, 205), (1050, 760, 410), '#534634', category='Terrain')
        for y in (745, 1555):
            b.box('Raised bed side', (x, y, 225), (1140, 55, 450), '#96714a', material='wood')
        for dx in (-550, 550):
            b.box('Raised bed end', (x + dx, 1150, 225), (55, 820, 450), '#ab8657', material='wood')
        for dx in (-320, 0, 320):
            for dy in (-150, 140):
                root = np.array([x + dx, 1150 + dy, 410.])
                for branch in range(4):
                    a = branch * math.pi / 2 + dx / 300
                    tip = root + [math.cos(a) * 75, math.sin(a) * 75, 240 + branch * 35]
                    b.rod('Herb stem', root, tip, 3, '#56754b', category='Plant')
                    for t in (.55, .9):
                        leaf = root + (tip - root) * t
                        b.ellipsoid('Herb leaf', leaf, [85, 34, 9], ['#729354', '#557d4b', '#8a9d65'][branch % 3],
                                    quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)], category='Plant')
                if dx == 0:
                    top = root + [0, 0, 390]
                    b.rod('Flower stem', root, top, 3, '#5f7d4c', category='Plant')
                    b.ellipsoid('Flower center', top, [18, 18, 12], '#c3a64d', category='Plant')
                    for petal in range(5):
                        a = petal * math.tau / 5
                        b.ellipsoid('Flower petal', top + [math.cos(a) * 26, math.sin(a) * 26, 0], [24, 16, 7], '#d1b5ce',
                                    quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)], category='Plant')
    b.table('Garden bench', 900, -1250, 450, 1500, 430, '#98734f')
    b.box('Bench back', (900, -1460, 720), (1500, 55, 440), '#ae8754', material='wood')
    for z in (565, 695, 825):
        b.box('Bench back seam', (900, -1490, z), (1470, 4, 12), '#594d39')
    b.plant(-1600, -1000, 0, 1.2); b.plant(-1050, -1200, 0, .8)
    b.cylinder('Watering can', (-600, -1100, 160), 125, 300, '#718d80', material='metal')
    b.rod('Watering spout', (-500, -1100, 150), (-250, -1100, 340), 25, '#718d80', material='metal')
    rng = np.random.default_rng(23)
    for i in range(30):
        x, y = float(rng.uniform(-2150, 2150)), float(rng.uniform(240, 580))
        b.ellipsoid('Pebble', (x, y, 12), [float(rng.uniform(12, 28)), 14, 12], '#92907a', category='Terrain')
        for j in range(3):
            b.rod('Grass blade', (x + 40, y, 0), (x + 30 + j * 10, y + 8, 40 + j * 15), 1.5, '#637d45', category='Plant')
    b.crumbs(8, -95, 15)
    return b.finish()


@lru_cache(maxsize=5)
def get_scene(key='lab'):
    if key == 'lab':
        return Builder('lab', 'Laboratory', 'Open calibration arena with a grid and controlled sensory cue.',
                       [200, 200, 20], [0, 0, 0], 'Calibration floor', '#c3cdd0', '#c3cdd0', [12, 3, .01], [4, 2, 1.5]).finish()
    factory = {'kitchen': kitchen, 'living-room': living_room, 'bedroom': bedroom, 'garden': garden}.get(key)
    if factory is None:
        raise ValueError('Unknown scene')
    return factory()


def scene_summary(key):
    scene = get_scene(key)
    return {k: v for k, v in scene.items() if k != 'objects'} | {'solid_count': len(scene['objects'])}
