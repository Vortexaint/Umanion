# Empty Bar Color RGB: 118, 117, 118

RACES_COUNT = {
    'Turf': 31,
    'Dirt': 13,
    'Sprint': 6,
    'Mile': 14,
    'Medium': 20,
    'Long': 4,
}


def RacesCount():
    return RACES_COUNT.copy()


def UmaCheck(track_affinity=None, distance_affinity=None, style_affinity=None):
    tracks = []
    distances = []
    styles = []

    if track_affinity:
        for name, val in track_affinity.items():
            if val == 'A':
                tracks.append(name)

    if distance_affinity:
        for name, val in distance_affinity.items():
            if val == 'A':
                distances.append(name)

    if style_affinity:
        for name, val in style_affinity.items():
            if val == 'A':
                styles.append(name)

    return {'tracks': tracks, 'distances': distances, 'styles': styles}


def CurrentUma(uma_result):
    return uma_result


def StyleCondition(style_name):
    s = style_name.strip().lower()
    if s == 'front':
        return {'Speed': 750, 'Power': 350, 'Guts': 300, 'Wit': 300}
    if s == 'mile':
        return {'Speed': 650, 'Power': 500, 'Guts': 300, 'Wit': 300}
    if s == 'late':
        return {'Speed': 550, 'Power': 600, 'Guts': 300, 'Wit': 300}
    if s == 'end':
        return {'Speed': 550, 'Power': 600, 'Guts': 300, 'Wit': 300}
    return {}


def DistanceCondition(junior=False, classic=False, senior=False):
    # Return stamina thresholds
    if senior:
        return {'Sprint': 300, 'Mile': 400, 'Medium': 450, 'Long': 550}
    if junior and classic:
        return {'Sprint': 150, 'Mile': 200, 'Medium': 250, 'Long': 300}
    return {'Sprint': 150, 'Mile': 200, 'Medium': 250, 'Long': 300}


if __name__ == '__main__':
    print('Races count:', RacesCount())
    demo = UmaCheck(
        track_affinity={'Turf': 'A', 'Dirt': 'B'},
        distance_affinity={'Sprint': 'A', 'Mile': 'A'},
        style_affinity={'Front': 'A'}
    )