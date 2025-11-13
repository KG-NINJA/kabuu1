def _prepare_training_frame(self, history: pd.DataFrame) -> pd.DataFrame:
        """特徴量を追加し、欠損値を除外する"""
        logging.info(f"元のデータ行数: {len(history)}")
        logging.info(f"利用可能なカラム: {list(history.columns)}")
        
        augmented = self._augment_with_features(history)
        logging.info(f"特徴量追加後のカラム: {list(augmented.columns)}")
        
        # 特徴量が正しく計算されたか確認
        for col in self.feature_columns:
            if col not in augmented.columns:
                logging.error(f"必須カラム '{col}' が見つかりません")
                return pd.DataFrame()
        
        augmented["target"] = augmented["close"].shift(-1)
        
        # dropna前に状態を確認
        logging.info(f"dropna前のデータ行数: {len(augmented)}")
        logging.info(f"各カラムのNull数:")
        for col in self.feature_columns + ["target"]:
            null_count = augmented[col].isnull().sum()
            logging.info(f"  {col}: {null_count} / {len(augmented)}")
        
        # 検査対象のカラムのみ存在するか確認
        subset_cols = [col for col in self.feature_columns + ["target"] 
                       if col in augmented.columns]
        
        if not subset_cols:
            logging.error("dropnaに渡すカラムが存在しません")
            return pd.DataFrame()
        
        try:
            augmented = augmented.dropna(subset=subset_cols)
            logging.info(f"dropna後のデータ行数: {len(augmented)}")
            
            if len(augmented) < 5:
                logging.warning(f"学習データが不足しています（{len(augmented)} 行）。最低5行は必要です。")
            
            return augmented
        except KeyError as e:
            logging.error(f"dropnaでKeyError: {e}")
            logging.error(f"期待していたカラム: {self.feature_columns + ['target']}")
            logging.error(f"実際のカラム: {list(augmented.columns)}")
            return pd.DataFrame()
