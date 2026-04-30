import pandas as pd
import json

# Carregar os dois CSVs
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

print(f"Total de filmes: {len(movies)}")
print(f"Total de créditos: {len(credits)}")
print(f"\nColunas em movies: {list(movies.columns)}")
print(f"\nColunas em credits: {list(credits.columns)}")

# Ver um filme exemplo
print("\n=== EXEMPLO: Primeiro filme ===")
first_movie = movies.iloc[0]
print(f"Título: {first_movie['title']}")
print(f"Sinopse: {first_movie['overview'][:200]}...")
print(f"Géneros (raw): {first_movie['genres'][:200]}")

# Os géneros vêm como string JSON, vamos decodificar
genres = json.loads(first_movie['genres'])
print(f"Géneros (parsed): {[g['name'] for g in genres]}")

# Ver o cast desse mesmo filme
movie_id = first_movie['id']
movie_credits = credits[credits['movie_id'] == movie_id].iloc[0]
cast = json.loads(movie_credits['cast'])[:5]  # top 5 atores
print(f"\nTop 5 atores: {[c['name'] for c in cast]}")

# Ver realizador
crew = json.loads(movie_credits['crew'])
director = [c['name'] for c in crew if c['job'] == 'Director']
print(f"Realizador: {director}")
