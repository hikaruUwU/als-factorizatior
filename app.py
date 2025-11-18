import os

from enviroment import DB_URI

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pandas as pd
from sqlalchemy import create_engine

import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

from tqdm import tqdm

engine = create_engine(DB_URI, pool_recycle=3600)

def get_data_from_mysql_sqlalchemy():
    sql = """
    SELECT user, item, times AS rating 
    FROM suki;
    """
    try:
        data = pd.read_sql(sql, engine)
        return data
    except Exception as ignored:
        return pd.DataFrame()

df = get_data_from_mysql_sqlalchemy()

if df.empty:
    print("Cannot fetch data from database.")
    exit()

df.columns = ['user', 'item', 'rating']

user_to_index = {user: i for i, user in enumerate(df['user'].unique())}
item_to_index = {item: i for i, item in enumerate(df['item'].unique())}

df['user_index'] = df['user'].map(user_to_index)
df['item_index'] = df['item'].map(item_to_index)

index_to_item = {i: item for item, i in item_to_index.items()}
# 增加索引到用户ID的映射 (用于批量导出)
index_to_user = {i: user for user, i in user_to_index.items()}

N_USERS = len(user_to_index)
N_ITEMS = len(item_to_index)

user_item_matrix = sparse.csr_matrix(
    (df['rating'].astype(float),
     (df['user_index'], df['item_index'])),
    shape=(N_USERS, N_ITEMS)
)

print("Data is ready to train.")

data_weighted = bm25_weight(user_item_matrix, K1=100, B=0.8)

item_user_matrix = data_weighted.T.tocsr()

NUM_FACTORS = 64
TOP_K_RECO = 100

model = AlternatingLeastSquares(
    factors=NUM_FACTORS,
    regularization=0.01,
    alpha=20,
    iterations=20
)
model.fit(item_user_matrix)
print("\nTrained.")

output_dir = 'recommendation_data'
os.makedirs(output_dir, exist_ok=True)
recommendation_list = []

print(f"Generating TOP {TOP_K_RECO} recommendations.")

for user_index in tqdm(range(N_USERS), desc="Calculating User Recommendations"):

    original_user_id = index_to_user[user_index]
    single_user_row = user_item_matrix.getrow(user_index)

    recommendations_index, scores = model.recommend(
        userid=user_index,
        user_items=single_user_row,
        N=TOP_K_RECO,
        filter_already_liked_items=True
    )

    top_items = [index_to_item[idx] for idx in recommendations_index]

    row_data = [original_user_id] + top_items

    if len(top_items) < TOP_K_RECO:
        row_data += [None] * (TOP_K_RECO - len(top_items))

    recommendation_list.append(row_data)

print("\nRecommendation generation complete.")

# --- 导出特征矩阵 (保持不变) ---
user_factors = model.user_factors
df_user_factors = pd.DataFrame(user_factors)
df_user_factors.index = user_to_index.keys()
df_user_factors.index.name = 'user_id'
user_factors_path = os.path.join(output_dir, 'user_factors.csv')
df_user_factors.to_csv(user_factors_path, header=False, index=True, float_format='%.6f')

item_factors = model.item_factors
df_item_factors = pd.DataFrame(item_factors)
df_item_factors.index = item_to_index.keys()
df_item_factors.index.name = 'item_id'
item_factors_path = os.path.join(output_dir, 'item_factors.csv')
df_item_factors.to_csv(item_factors_path, header=False, index=True, float_format='%.6f')
print("Matrix are exported.")

header = ['user_id'] + [f'reco_{i + 1}' for i in range(TOP_K_RECO)]
df_recommendations = pd.DataFrame(recommendation_list, columns=header)
recommendation_csv_path = os.path.join(output_dir, f'recommendations_top_{TOP_K_RECO}.csv')
df_recommendations.to_csv(recommendation_csv_path, index=False, header=True)

print(f"Recommendations exported.")