# train_and_submit.py
import os, argparse, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
import time

def load_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir,'train.csv'))
    test = pd.read_csv(os.path.join(data_dir,'test.csv'))
    books = pd.read_csv(os.path.join(data_dir,'books.csv'))
    users = pd.read_csv(os.path.join(data_dir,'users.csv'))
    # optional files
    return train, test, books, users

def prepare_temporal_holdout(train):
    # only interactions where has_read==1 and rating>0 matter for target
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    reads = train[(train['has_read']==1) & (train['rating']>0)].copy()
    reads = reads.sort_values(['user_id','timestamp'])
    # last per user -> validation
    last = reads.groupby('user_id').tail(1)
    train_tr = reads.drop(last.index)
    val = last.copy()
    # If a user had only one read, they will not be in train_tr; that's fine.
    return train_tr.reset_index(drop=True), val.reset_index(drop=True)

def build_aggregates(train_tr):
    global_mean = train_tr['rating'].mean()
    user_agg = train_tr.groupby('user_id')['rating'].agg(['mean','count','std']).reset_index().rename(
        columns={'mean':'user_mean','count':'user_count','std':'user_std'})
    book_agg = train_tr.groupby('book_id')['rating'].agg(['mean','count','std']).reset_index().rename(
        columns={'mean':'book_mean','count':'book_count','std':'book_std'})
    return global_mean, user_agg, book_agg

def merge_features(df, user_agg, book_agg, books, users, global_mean):
    df = df.merge(user_agg, on='user_id', how='left')
    df = df.merge(book_agg, on='book_id', how='left')
    df = df.merge(books[['book_id','publication_year','author_id','avg_rating','language','publisher']], on='book_id', how='left')
    df = df.merge(users, on='user_id', how='left')
    # fill na
    df['user_mean'].fillna(global_mean, inplace=True)
    df['book_mean'].fillna(global_mean, inplace=True)
    df['user_count'].fillna(0, inplace=True)
    df['book_count'].fillna(0, inplace=True)
    df['user_std'].fillna(0, inplace=True)
    df['book_std'].fillna(0, inplace=True)
    df['avg_rating'].fillna(global_mean, inplace=True)
    df['publication_year'].fillna(0, inplace=True)
    df['author_id'].fillna(-1, inplace=True)
    df['publisher'].fillna(-1, inplace=True)
    df['language'].fillna(-1, inplace=True)
    df['age'].fillna(-1, inplace=True)
    df['gender'].fillna(-1, inplace=True)
    return df

def add_target_encoding(train_tr, val, test_feat, col, target='rating', min_count=20, alpha=10):
    # simple regularized target encoding using train_tr
    stats = train_tr.groupby(col)[target].agg(['mean','count']).reset_index().rename(columns={'mean':f'{col}_te_mean','count':f'{col}_te_count'})
    stats[f'{col}_te'] = (stats[f'{col}_te_mean']*stats[f'{col}_te_count'] + train_tr[target].mean()*alpha) / (stats[f'{col}_te_count'] + alpha)
    for df in [val, test_feat]:
        df = df.merge(stats[[col,f'{col}_te']], on=col, how='left')
        df[f'{col}_te'].fillna(train_tr[target].mean(), inplace=True)
    # For training set we also add (for completeness) - but not used in our simplistic pipeline
    train_tr = train_tr.merge(stats[[col,f'{col}_te']], on=col, how='left')
    train_tr[f'{col}_te'].fillna(train_tr[target].mean(), inplace=True)
    return train_tr, val, test_feat

def make_features(train_tr, val, test, books, users):
    global_mean, user_agg, book_agg = build_aggregates(train_tr)
    train_tr = merge_features(train_tr, user_agg, book_agg, books, users, global_mean)
    val = merge_features(val, user_agg, book_agg, books, users, global_mean)
    test_feat = merge_features(test.copy(), user_agg, book_agg, books, users, global_mean)
    # add simple engineered features
    for df in [train_tr, val, test_feat]:
        df['pub_age'] = (2025 - df['publication_year']).clip(lower=0)  # approximate age
        df['user_activity'] = df['user_count']
        df['book_popularity'] = df['book_count']
        # interaction feature
        df['user_book_mean_diff'] = df['user_mean'] - df['book_mean']
    # target-encode author_id and publisher using train_tr
    train_tr, val, test_feat = add_target_encoding(train_tr, val, test_feat, 'author_id')
    train_tr, val, test_feat = add_target_encoding(train_tr, val, test_feat, 'publisher')
    # label-encode categorical small-cardinality
    for col in ['language','gender']:
        le = LabelEncoder()
        # fit on concatenation to avoid unseen issues
        all_vals = pd.concat([train_tr[col].astype(str), val[col].astype(str), test_feat[col].astype(str)])
        le.fit(all_vals)
        for df in [train_tr, val, test_feat]:
            df[col] = le.transform(df[col].astype(str))
    features = [
        'user_mean','user_count','user_std','book_mean','book_count','book_std',
        'avg_rating','publication_year','pub_age','author_id_te','publisher_te',
        'language','age','gender','user_book_mean_diff','user_activity','book_popularity'
    ]
    # ensure columns exist
    for df in [train_tr, val, test_feat]:
        for f in features:
            if f not in df.columns:
                df[f] = 0
    return train_tr, val, test_feat, features

def train_model(X_train, y_train, X_val, y_val):
    # try lightgbm first
    try:
        import lightgbm as lgb
        params = {'objective':'regression','metric':'rmse','learning_rate':0.05,'num_leaves':31,'seed':42,'verbose':-1}
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dtrain,dval], early_stopping_rounds=50, verbose_eval=100)
        use = 'lgb'
    except Exception as e:
        print("LightGBM unavailable or failed, fallback to sklearn HistGradientBoosting. Error:", e)
        model = HistGradientBoostingRegressor(random_state=42, max_iter=500, learning_rate=0.05)
        model.fit(X_train, y_train)
        use = 'sklearn_hgb'
    return model, use

def main(data_dir, out_path):
    train, test, books, users = load_data(data_dir)
    train_tr, val = prepare_temporal_holdout(train)
    print(f"Train interactions for training: {len(train_tr)}, validation interactions: {len(val)}, test rows: {len(test)}")
    train_tr, val, test_feat, features = make_features(train_tr, val, test, books, users)
    X_train = train_tr[features]
    y_train = train_tr['rating']
    X_val = val[features]
    y_val = val['rating']
    print("Feature count:", len(features))
    # quick baseline: regularized user+item bias
    global_mean = y_train.mean()
    reg = 10
    u = train_tr.groupby('user_id')['rating'].agg(['sum','count']).reset_index()
    u['u_bias'] = (u['sum'] - u['count']*global_mean)/(u['count']+reg)
    b = train_tr.groupby('book_id')['rating'].agg(['sum','count']).reset_index()
    b['b_bias'] = (b['sum'] - b['count']*global_mean)/(b['count']+reg)
    val_bias = val.merge(u[['user_id','u_bias']], on='user_id', how='left').merge(b[['book_id','b_bias']], on='book_id', how='left')
    val_bias['u_bias'].fillna(0,inplace=True); val_bias['b_bias'].fillna(0,inplace=True)
    bias_pred = global_mean + val_bias['u_bias'] + val_bias['b_bias']
    print("Baseline bias RMSE:", mean_squared_error(y_val, bias_pred, squared=False), "R2:", r2_score(y_val,bias_pred))
    # train model
    print("Training model...")
    model, engine = train_model(X_train, y_train, X_val, y_val)
    if engine == 'lgb':
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    else:
        val_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    r2 = r2_score(y_val, val_pred)
    print(f"Validation RMSE: {rmse:.6f}, R2: {r2:.6f}  (engine: {engine})")
    # predict test
    X_test = test_feat[features]
    if engine == 'lgb':
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    else:
        test_pred = model.predict(X_test)
    # clip preds to rating range if known (e.g., 1..10). Here many ratings are >0; check train min/max
    rmin, rmax = train_tr['rating'].min(), train_tr['rating'].max()
    test_pred = np.clip(test_pred, rmin, rmax)
    submission = pd.DataFrame({'user_id': test['user_id'], 'book_id': test['book_id'], 'rating': test_pred})
    submission.to_csv(out_path, index=False)
    print("Saved submission to", out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.', help='path to folder with train.csv,test.csv,books.csv,users.csv')
    parser.add_argument('--out', dest='out', type=str, default='submission.csv')
    args = parser.parse_args()
    start = time.time()
    main(args.data_dir, args.out)
    print("Done in %.1fs" % (time.time()-start))
