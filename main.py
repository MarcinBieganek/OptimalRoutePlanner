
import pandas as pd
import itertools

def load_attractions(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    return df

def load_distance_matrix(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = df.columns.str.strip()
    df.index = df.index.str.strip()
    return df

def selected_matric(dists, selected_attractions, start_point):
    selected_dists = dists.loc[selected_attractions, selected_attractions]
    selected_dists = reorder_matrix(selected_dists, start_point)

    return selected_dists

def get_visit_times(attractions, selected_dists):
    time_dict = dict(zip(attractions['nazwa'], attractions['czas_min']))
    return [time_dict.get(place, None) for place in selected_dists.index]

def reorder_matrix(df, start_point):
    if start_point not in df.index:
        raise ValueError(f"🔴 Punkt '{start_point}' nie istnieje w macierzy.")
    
    # Przesuń wybrany punkt na początek
    labels = list(df.index)
    labels.remove(start_point)
    labels.insert(0, start_point)

    df = df.loc[labels, labels]
    return df


def held_karp_with_limit(time, visit_time, max_time):
    n = len(time)
    C = {}

    # Inicjalizacja: tylko jedna atrakcja odwiedzona
    for k in range(1, n):
        C[(1 << k, k)] = (visit_time[0] + time[0][k] + visit_time[k], 0)

    best_path = []
    best_time = float('inf')
    best_visited = 0

    # Budujemy wyniki dynamicznie
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = sum(1 << x for x in subset)

            for k in subset:
                prev_bits = bits & ~(1 << k)

                min_cost = float('inf')
                prev_node = None

                for m in subset:
                    if m == k:
                        continue
                    if (prev_bits, m) in C:
                        cost = C[(prev_bits, m)][0] + time[m][k] + visit_time[k]
                        if cost < min_cost:
                            min_cost = cost
                            prev_node = m

                if prev_node is not None:
                    C[(bits, k)] = (min_cost, prev_node)

    # Sprawdzamy wszystkie możliwe końcowe stany
    for (bits, k), (cost, prev) in C.items():
        total_time = cost + time[k][0]  # Dodaj powrót do startu
        visited = bin(bits).count('1') + 1  # +1 za punkt startowy

        if total_time <= max_time and visited > best_visited:
            best_visited = visited
            best_time = total_time

            # Odtwarzamy trasę
            path = [0]
            last = k
            b = bits
            for _ in range(visited - 1):
                path.append(last)
                _, prev = C[(b, last)]
                b &= ~(1 << last)
                last = prev
            path.append(0)
            best_path = path[::-1]

    if not best_path:
        return None, []

    return best_time, best_path


def nearest_neighbor_with_reserved_return(matrix, visit_times, minutes_per_km, start_index=0, max_time=360):
    n = len(matrix)
    visited = [False] * n
    path = [start_index]
    total_time = visit_times[start_index]
    total_distance = 0
    current = start_index
    visited[current] = True

    while True:
        nearest = None
        min_dist = float('inf')

        for i in range(n):
            if not visited[i]:
                travel_time = matrix[current, i] * minutes_per_km
                return_time = matrix[i, start_index] * minutes_per_km
                projected_time = total_time + travel_time + visit_times[i]
                full_projected_time = projected_time + return_time

                if full_projected_time <= max_time and matrix[current, i] < min_dist:
                    min_dist = matrix[current, i]
                    nearest = i

        if nearest is None:
            break

        path.append(nearest)
        visited[nearest] = True
        distance_added = matrix[current, nearest]
        time_added = distance_added * minutes_per_km + visit_times[nearest]
        total_distance += distance_added
        total_time += time_added
        current = nearest

    # Powrót do startu
    return_distance = matrix[current, start_index]
    return_time = return_distance * minutes_per_km
    total_distance += return_distance
    total_time += return_time
    path.append(start_index)

    return path, total_distance, total_time


def format_time(minutes):
    return f"{minutes // 60}h {minutes % 60}min"

def pretty_print_results(name, path, total_time, selected_dists):
    s = list(selected_dists.index)

    print(f"--- {name} ---")
    print("Trasa:")
    for idx in path:
        print(f"  {idx}: {s[idx]}")
    print(f"Liczba odwiedzonych atrakcji: {len(set(path))}")
    print(f"Łączny czas: {format_time(int(total_time))}")

    print()

def test_route(attractions, dists, start_point, selected, average_speed_kmh, max_time):
    # predkosc w minutach
    minutes_per_km = 60 / average_speed_kmh

    # ustawienie kolejności tak, by punkt początkowy był pierwszy
    dists = reorder_matrix(dists, start_point)

    # ograniczenie macierzy do wybranych punktów
    selected_dists = selected_matric(dists, selected, start_point)
    selected_times = selected_dists * minutes_per_km
    times_matrix = selected_times.to_numpy()
    dists_matrix = selected_dists

    # zapisanie czasu zwiedzania w odpowiedniej kolejności
    visit_times = get_visit_times(attractions, selected_dists)

    # najlepsza trasa zwiedzania algorytmem najbliższego sąsiada
    path, total_km, total_time = nearest_neighbor_with_reserved_return(dists_matrix.values, visit_times, minutes_per_km, start_index=0, max_time=max_time)

    pretty_print_results("Heurystka Sąsiadów", path, total_time, selected_dists)

    # najlepsza trasa zwiedzania zmodyfikowanym algorytmem helda-karpa
    best_time, best_path = held_karp_with_limit(times_matrix, visit_times, max_time)

    pretty_print_results("Helda-Karpa", best_path, best_time, selected_dists)

def main():
    # Zmienne
    csv_dist_path = "dist.csv"
    csv_attractions_path = "attractions.csv"
    #start_point = input("🔰 Podaj nazwę punktu początkowego (dokładnie jak w pliku CSV): ").strip()
    #selected = ['Watykan', 'Koloseum', 'Panteon', 'Fontanna di Trevi', 'Villa Borghese', 'Katakumby św. Kaliksta']
    average_speed_kmh = 18
    minutes_per_km = 60 / average_speed_kmh
    max_time = 420 #(6h dałam)

    # wczytanie danych
    attractions = load_attractions(csv_attractions_path)
    dists = load_distance_matrix(csv_dist_path)

    #test_route(attractions, dists, start_point, selected, average_speed_kmh, max_time)

    # test case 1
    start_point = 'Watykan'
    selected = [
        "Watykan",
        "Bazylika Świętego Piotra",
        "Muzea Watykańskie",
        "Kaplica Sykstyńska",
        "Zamek Świętego Anioła",
        "Panteon",
        "Fontanna di Trevi",
        "Plac Hiszpański i Schody Hiszpańskie",
        "Piazza Navona",
        "Campo de' Fiori",
        "Koloseum",
        "Forum Romanum"
    ]

    test_route(attractions, dists, start_point, selected, average_speed_kmh, max_time)

    # test case 2
    start_point = 'Forum Romanum'
    selected = [
        "Ołtarz Ojczyzny (Vittoriano)",
        "Bazylika Santa Maria Maggiore",
        "Bazylika św. Jana na Lateranie",
        "Zamek Świętego Anioła",
        "Panteon",
        "Fontanna di Trevi",
        "Plac Hiszpański i Schody Hiszpańskie",
        "Piazza Navona",
        "Campo de' Fiori",
        "Koloseum",
        "Forum Romanum"
    ]

    test_route(attractions, dists, start_point, selected, average_speed_kmh, max_time)

    # test case 3
    start_point = 'Koloseum'
    selected = [
        "Watykan",
        "Bazylika Świętego Piotra",
        "Muzea Watykańskie",
        "Kaplica Sykstyńska",
        "Zamek Świętego Anioła",
        "Panteon",
        "Fontanna di Trevi",
        "Plac Hiszpański i Schody Hiszpańskie",
        "Piazza Navona",
        "Campo de' Fiori",
        "Koloseum",
        "Forum Romanum",
        "Palatyn",
        "Łuk Konstantyna",
        "Kapitol",
        "Ołtarz Ojczyzny (Vittoriano)",
        "Bazylika Santa Maria Maggiore",
        "Bazylika św. Jana na Lateranie",
        "Katakumby św. Kaliksta",
        "Villa Borghese",
        "Piazza del Popolo"
    ]

    test_route(attractions, dists, start_point, selected, average_speed_kmh, max_time)

    



if __name__ == "__main__":
    main()