@st.cache_resource
def load_and_train_models(file):
    df = pd.read_csv(file)
    df = remove_outliers_iqr(df, df.select_dtypes(include=['int64', 'float64']).columns)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=SEED),
        'Random Forest': RandomForestClassifier(random_state=SEED),
        'SVM': SVC(probability=True, random_state=SEED),
        'Naive Bayes': GaussianNB(),
        'Ridge Classifier': RidgeClassifier(random_state=SEED),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED)
    }

    results = []

    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        recall = recall_score(y_test, y_pred)
        results.append((name, recall, pipeline))

    # Neural Network
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    keras_model = build_keras_model(X_train_processed.shape[1])
    keras_model.fit(X_train_processed, y_train, epochs=50, validation_split=0.2, verbose=0)
    y_pred_keras = (keras_model.predict(X_test_processed) > 0.5).astype("int32")
    keras_recall = recall_score(y_test, y_pred_keras)
    results.append(("Neural Network", keras_recall, (preprocessor, keras_model)))

    results.sort(key=lambda x: x[1], reverse=True)
    top3 = results[:3]

    return top3, numeric_features, categorical_features, X
