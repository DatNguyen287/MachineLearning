import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PHẦN 1: CÁC THUẬT TOÁN BASE MODELS
# ============================================================================

# --- 1.1 RIDGE REGRESSION ---
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.w = None
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def fit(self, X, y):
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.X_mean) / self.X_std
        
        self.y_mean = np.mean(y)
        self.y_std = np.std(y) + 1e-8
        y_norm = (y - self.y_mean) / self.y_std
        
        Xb = add_bias(X_norm)
        n = Xb.shape[1]
        I = np.eye(n)
        I[0, 0] = 0
        
        A = Xb.T @ Xb + (self.alpha + 1e-4) * I
        b = Xb.T @ y_norm
        
        try:
            self.w = np.linalg.solve(A, b)
            if np.any(np.isnan(self.w)) or np.any(np.isinf(self.w)):
                raise np.linalg.LinAlgError("NaN in weights")
        except np.linalg.LinAlgError:
            try:
                self.w = np.linalg.lstsq(A, b, rcond=1e-10)[0]
                if np.any(np.isnan(self.w)):
                    self.w = np.zeros(n)
                    self.w[0] = self.y_mean
            except:
                self.w = np.zeros(n)
                self.w[0] = self.y_mean

    def predict(self, X):
        if self.w is None:
            return np.zeros(len(X))
        X_norm = (X - self.X_mean) / self.X_std
        y_pred_norm = add_bias(X_norm) @ self.w
        return y_pred_norm * self.y_std + self.y_mean


# --- 1.2 RANDOM FOREST ---
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegressor:
    def __init__(self, min_samples_split=10, max_depth=6, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        if self.n_features is None:
            self.n_features = int(np.sqrt(X.shape[1]))
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return Node(value=np.mean(y))
        
        feature_idxs = np.random.choice(X.shape[1], self.n_features, replace=False)
        best_split = self._get_best_split(X, y, feature_idxs)
        
        if best_split['var_red'] <= 0:
            return Node(value=np.mean(y))
        
        left_tree = self._grow_tree(best_split['X_left'], best_split['y_left'], depth + 1)
        right_tree = self._grow_tree(best_split['X_right'], best_split['y_right'], depth + 1)
        
        return Node(best_split['feature_idx'], best_split['threshold'], left_tree, right_tree)

    def _get_best_split(self, X, y, feature_indices):
        best_split = {'var_red': -1}
        for feat_idx in feature_indices:
            thresholds = np.percentile(X[:, feat_idx], np.linspace(10, 90, 10))
            for threshold in thresholds:
                left_mask = X[:, feat_idx] <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                y_left, y_right = y[left_mask], y[right_mask]
                var_red = np.var(y) - (len(y_left)/len(y) * np.var(y_left) + len(y_right)/len(y) * np.var(y_right))
                if var_red > best_split['var_red']:
                    best_split = {
                        'feature_idx': feat_idx,
                        'threshold': threshold,
                        'X_left': X[left_mask],
                        'y_left': y_left,
                        'X_right': X[right_mask],
                        'y_right': y_right,
                        'var_red': var_red
                    }
        return best_split

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

class RandomForestRegressor:
    def __init__(self, n_trees=100, max_depth=6, min_samples_split=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def fit(self, X, y):
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.X_mean) / self.X_std
        
        self.y_mean = np.mean(y)
        self.y_std = np.std(y) + 1e-8
        y_norm = (y - self.y_mean) / self.y_std
        
        self.trees = []
        for i in range(self.n_trees):
            idxs = np.random.choice(len(X_norm), len(X_norm), replace=True)
            tree = DecisionTreeRegressor(self.min_samples_split, self.max_depth)
            tree.fit(X_norm[idxs], y_norm[idxs])
            self.trees.append(tree)

    def predict(self, X):
        X_norm = (X - self.X_mean) / self.X_std
        predictions = np.array([tree.predict(X_norm) for tree in self.trees])
        y_pred_norm = np.mean(predictions, axis=0)
        return y_pred_norm * self.y_std + self.y_mean


# --- 1.3 GRADIENT BOOSTING ---
class GradientBoosting:
    def __init__(self, learning_rate=0.1, n_trees=100, max_depth=3):
        self.lr = learning_rate
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        
    def fit(self, X, y):
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.X_mean) / self.X_std
        
        self.y_mean = np.mean(y)
        self.y_std = np.std(y) + 1e-8
        y_norm = (y - self.y_mean) / self.y_std
        
        y_pred = np.zeros_like(y_norm)
        
        for tree_idx in range(self.n_trees):
            residuals = y_norm - y_pred
            tree = self._build_tree(X_norm, residuals, depth=0)
            self.trees.append(tree)
            tree_preds = self._predict_tree(tree, X_norm)
            y_pred += self.lr * tree_preds
    
    def _build_tree(self, X, residuals, depth=0):
        n_samples = X.shape[0]
        if depth >= self.max_depth or n_samples < 10:
            return {'type': 'leaf', 'value': np.mean(residuals)}
        
        best_gain = -np.inf
        best_split = None
        parent_var = np.var(residuals)
        
        for feat_idx in range(X.shape[1]):
            thresholds = np.percentile(X[:, feat_idx], [25, 50, 75])
            for threshold in thresholds:
                left_mask = X[:, feat_idx] <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) < 5 or np.sum(right_mask) < 5:
                    continue
                left_var = np.var(residuals[left_mask])
                right_var = np.var(residuals[right_mask])
                gain = parent_var - (np.sum(left_mask) * left_var + np.sum(right_mask) * right_var) / n_samples
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feat_idx': feat_idx,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }
        
        if best_split is None or best_gain <= 0:
            return {'type': 'leaf', 'value': np.mean(residuals)}
        
        left_tree = self._build_tree(X[best_split['left_mask']], residuals[best_split['left_mask']], depth + 1)
        right_tree = self._build_tree(X[best_split['right_mask']], residuals[best_split['right_mask']], depth + 1)
        
        return {
            'type': 'split',
            'feat_idx': best_split['feat_idx'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }
    
    def _predict_tree(self, tree, X):
        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            node = tree
            while node['type'] != 'leaf':
                if X[i, node['feat_idx']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            predictions[i] = node['value']
        return predictions
    
    def predict(self, X):
        X_norm = (X - self.X_mean) / self.X_std
        y_pred = np.zeros(X_norm.shape[0])
        for tree in self.trees:
            y_pred += self.lr * self._predict_tree(tree, X_norm)
        return y_pred * self.y_std + self.y_mean


# --- 1.4 KNN ---
class KNNRegressor:
    def __init__(self, n_neighbors=10):
        self.k = n_neighbors
        self.X_train = None
        self.y_train = None
        self.X_mean = None
        self.X_std = None
        
    def fit(self, X, y):
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-8
        self.X_train = (X - self.X_mean) / self.X_std
        self.y_train = y
    
    def predict(self, X):
        X_norm = (X - self.X_mean) / self.X_std
        predictions = []
        for test_point in X_norm:
            distances = np.sqrt(np.sum((self.X_train - test_point)**2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            weights = 1 / (distances[k_indices] + 1e-8)
            predictions.append(np.sum(weights * self.y_train[k_indices]) / np.sum(weights))
        return np.array(predictions)


# ============================================================================
# PHẦN 2: STACKING ENSEMBLE PREDICTOR
# ============================================================================

class NYAStackingPredictor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.price_test = None
        self.models = {}
        self.meta_model = None
        self.predictions = {}
        
    def load_and_filter_data(self):
        df = pd.read_csv(self.file_path)
        df = df[df['Index'] == 'NYA'].copy()
        print(f"Loaded {len(df)} rows")
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
        self.data = df
        return self
    
    def create_features(self):
        df = self.data.copy()
        df['Price_Current'] = df['Adj Close']
        df['Price_Next'] = df['Adj Close'].shift(-1)
        df['Target_Return'] = (df['Price_Next'] - df['Price_Current']) / df['Price_Current']
        
        df['Return_1d'] = df['Adj Close'].pct_change()
        df['Return_5d'] = df['Adj Close'].pct_change(5)
        for i in [1, 2, 3, 5, 10]:
            df[f'Return_Lag_{i}'] = df['Return_1d'].shift(i)
        
        close_prev = df['Adj Close'].shift(1)
        df['MA_5'] = close_prev.rolling(window=5).mean()
        df['MA_10'] = close_prev.rolling(window=10).mean()
        df['MA_20'] = close_prev.rolling(window=20).mean()
        df['Price_over_MA10'] = close_prev / (df['MA_10'] + 1e-8)
        
        df['Volatility_5'] = df['Return_1d'].rolling(window=5).std()
        df['Volatility_10'] = df['Return_1d'].rolling(window=10).std()
        df['Volatility_20'] = df['Return_1d'].rolling(window=20).std()
        
        df['Volume_MA_5'] = df['Volume'].shift(1).rolling(window=5).mean()
        df['Volume_ratio'] = df['Volume'] / (df['Volume_MA_5'] + 1e-8)
        
        df['Momentum_5'] = df['Adj Close'] - df['Adj Close'].shift(5)
        df['Momentum_10'] = df['Adj Close'] - df['Adj Close'].shift(10)
        df['Momentum_pct_5'] = df['Momentum_5'] / (df['Adj Close'].shift(5) + 1e-8)
        df['Momentum_pct_10'] = df['Momentum_10'] / (df['Adj Close'].shift(10) + 1e-8)
        df['Price_Range'] = (df['High'] - df['Low']) / (df['Low'] + 1e-8)
        
        rolling_high_10 = df['High'].rolling(window=10).max()
        rolling_low_10 = df['Low'].rolling(window=10).min()
        df['HL_position_10'] = (df['Close'] - rolling_low_10) / (rolling_high_10 - rolling_low_10 + 1e-8)
        
        rolling_high_20 = df['High'].rolling(window=20).max()
        rolling_low_20 = df['Low'].rolling(window=20).min()
        df['HL_position_20'] = (df['Close'] - rolling_low_20) / (rolling_high_20 - rolling_low_20 + 1e-8)
        
        delta = df['Adj Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        df = df.dropna()
        
        feature_cols = [c for c in df.columns if c.startswith(('Return', 'MA_', 'Price_', 'Vol', 'Momentum', 'HL', 'RSI'))]
        print(f"Created {len(feature_cols)} features, {len(df)} samples")
        
        self.data = df
        return self
    
    def prepare_data(self):
        exclude_cols = ['Date', 'Index', 'Close', 'Adj Close', 'Open', 'High', 'Low', 
                        'Volume', 'CloseUSD', 'Target_Return', 'Price_Current', 'Price_Next']
        X = self.data.drop(columns=[col for col in exclude_cols if col in self.data.columns])
        y = self.data['Target_Return']
        price_current = self.data['Price_Current']
        
        n_total = len(X)
        n_test = int(0.2 * n_total)
        n_val = int(0.2 * (n_total - n_test))
        
        print(f"Train: {n_total - n_test - n_val} | Val: {n_val} | Test: {n_test}")
        
        self.X_train = X.iloc[:-(n_test + n_val)].values
        self.y_train = y.iloc[:-(n_test + n_val)].values
        
        self.X_val = X.iloc[-(n_test + n_val):-n_test].values
        self.y_val = y.iloc[-(n_test + n_val):-n_test].values
        
        self.X_test = X.iloc[-n_test:].values
        self.y_test = y.iloc[-n_test:].values
        self.price_test = price_current.iloc[-n_test:].values
        
        return self
    
    def train_base_models(self):
        print("\nTraining base models...")
        
        self.models['Ridge'] = RidgeRegression(alpha=1.0)
        self.models['Ridge'].fit(self.X_train, self.y_train)
        
        self.models['RF'] = RandomForestRegressor(n_trees=200, max_depth=8, min_samples_split=5)
        self.models['RF'].fit(self.X_train, self.y_train)
        
        self.models['GradientBoosting'] = GradientBoosting(learning_rate=0.03, n_trees=300, max_depth=2)
        self.models['GradientBoosting'].fit(self.X_train, self.y_train)
        
        self.models['KNN'] = KNNRegressor(n_neighbors=10)
        self.models['KNN'].fit(self.X_train, self.y_train)
        
        print("Done.")
        return self
    
    def train_stacking_walkforward(self):
        print("\nTraining stacking model...")
        
        base_names = ['Ridge', 'RF', 'GradientBoosting', 'KNN']
        n_models = len(base_names)
        
        X_combined = np.vstack([self.X_train, self.X_val])
        y_combined = np.concatenate([self.y_train, self.y_val])
        
        min_train_size = int(0.55 * len(X_combined))
        step_size = int(0.05 * len(X_combined))
        
        wf_train_preds = []
        wf_train_actuals = []
        
        current_end = min_train_size
        n_windows = 0
        
        while current_end < len(X_combined):
            val_start = current_end
            val_end = min(current_end + step_size, len(X_combined))
            
            if val_end - val_start < 10:
                break
            
            X_wf_train = X_combined[:current_end]
            y_wf_train = y_combined[:current_end]
            X_wf_val = X_combined[val_start:val_end]
            y_wf_val = y_combined[val_start:val_end]
            
            window_preds = np.zeros((len(X_wf_val), n_models))
            
            for i, name in enumerate(base_names):
                if name == 'Ridge':
                    model = RidgeRegression(alpha=1.0)
                elif name == 'RF':
                    model = RandomForestRegressor(n_trees=200, max_depth=8, min_samples_split=5)
                elif name == 'GradientBoosting':
                    model = GradientBoosting(learning_rate=0.03, n_trees=300, max_depth=2)
                else:
                    model = KNNRegressor(n_neighbors=10)
                
                try:
                    model.fit(X_wf_train, y_wf_train)
                    window_preds[:, i] = model.predict(X_wf_val)
                except:
                    window_preds[:, i] = 0
            
            wf_train_preds.append(window_preds)
            wf_train_actuals.append(y_wf_val)
            
            n_windows += 1
            current_end += step_size
        
        wf_train_preds_all = np.vstack(wf_train_preds)
        wf_train_actuals_all = np.concatenate(wf_train_actuals)
        
        if np.any(np.isnan(wf_train_preds_all)):
            for col in range(wf_train_preds_all.shape[1]):
                col_data = wf_train_preds_all[:, col]
                if np.any(np.isnan(col_data)):
                    col_mean = np.nanmean(col_data) if not np.all(np.isnan(col_data)) else 0
                    wf_train_preds_all[np.isnan(col_data), col] = col_mean
        
        alpha_candidates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        n_meta_val = int(0.2 * len(wf_train_preds_all))
        meta_train_X = wf_train_preds_all[:-n_meta_val]
        meta_train_y = wf_train_actuals_all[:-n_meta_val]
        meta_val_X = wf_train_preds_all[-n_meta_val:]
        meta_val_y = wf_train_actuals_all[-n_meta_val:]
        
        best_alpha = 0.05
        best_score = float('inf')
        
        for alpha in alpha_candidates:
            meta_temp = RidgeRegression(alpha=alpha)
            meta_temp.fit(meta_train_X, meta_train_y)
            val_pred = meta_temp.predict(meta_val_X)
            rmse = np.sqrt(np.mean((meta_val_y - val_pred) ** 2))
            if rmse < best_score:
                best_score = rmse
                best_alpha = alpha
        
        self.meta_model = RidgeRegression(alpha=best_alpha)
        self.meta_model.fit(wf_train_preds_all, wf_train_actuals_all)
        
        test_preds = np.zeros((len(self.X_test), n_models))
        for i, name in enumerate(base_names):
            preds = self.models[name].predict(self.X_test)
            if np.any(np.isnan(preds)):
                preds = np.zeros(len(self.X_test))
            test_preds[:, i] = preds
        
        self.predictions['Stacking'] = self.meta_model.predict(test_preds)
        print("Done.")
        
        return self
    
    def evaluate_models(self):
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        results = []
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            self.predictions[name] = y_pred
            
            rmse = np.sqrt(np.mean((self.y_test - y_pred) ** 2))
            mae = np.mean(np.abs(self.y_test - y_pred))
            
            results.append({"Model": name, "RMSE": rmse, "MAE": mae})
        
        stack_pred = self.predictions['Stacking']
        stack_rmse = np.sqrt(np.mean((self.y_test - stack_pred) ** 2))
        stack_mae = np.mean(np.abs(self.y_test - stack_pred))
        results.append({"Model": "Stacking", "RMSE": stack_rmse, "MAE": stack_mae})
        
        print("\nReturn Prediction:")
        print(pd.DataFrame(results).to_string(index=False))
        
        print("\nPrice Prediction:")
        price_results = []
        for name in ['Ridge', 'RF', 'GradientBoosting', 'KNN', 'Stacking']:
            return_pred = self.predictions[name]
            price_pred = self.price_test * (1 + return_pred)
            price_actual = self.price_test * (1 + self.y_test)
            
            price_rmse = np.sqrt(np.mean((price_actual - price_pred) ** 2))
            price_mae = np.mean(np.abs(price_actual - price_pred))
            
            price_results.append({"Model": name, "RMSE": price_rmse, "MAE": price_mae})
        
        print(pd.DataFrame(price_results).to_string(index=False))
        print("=" * 80)
        
        return self


# ============================================================================
# PHẦN 3: CHẠY CHƯƠNG TRÌNH
# ============================================================================

if __name__ == "__main__":
    print("NYA Stock Prediction - Stacking Ensemble")
    print("=" * 80)
    
    predictor = NYAStackingPredictor('indexProcessed.csv')
    predictor.load_and_filter_data() \
             .create_features() \
             .prepare_data() \
             .train_base_models() \
             .train_stacking_walkforward() \
             .evaluate_models()
    
    # Lưu model
    with open('nya_model.pkl', 'wb') as f:
        pickle.dump(predictor, f)
    
    file_size = os.path.getsize('nya_model.pkl') / 1024 / 1024
    print(f"\nModel saved: nya_model.pkl ({file_size:.2f} MB)")
