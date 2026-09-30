import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
]


def run_clustering(
    df,
    X,
    output_dir,
    random_seed=3513,
    models_dir="models",
):
    output_dir = Path(output_dir)
    models_dir = Path(models_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Use ONLY the six input features.
    #
    # actual_yield_kg and dispatch_attention are deliberately
    # NOT included in clustering.
    # ---------------------------------------------------------

    X_features = X.astype(float)

    # ---------------------------------------------------------
    # Standardize features
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_features
    )

    # ---------------------------------------------------------
    # Evaluate k = 2, 3, 4, 5
    # ---------------------------------------------------------

    silhouette_scores = {}

    for k in range(2, 6):

        model = KMeans(
            n_clusters=k,
            random_state=random_seed,
            n_init=10,
        )

        labels = model.fit_predict(
            X_scaled
        )

        score = silhouette_score(
            X_scaled,
            labels,
        )

        silhouette_scores[str(k)] = float(
            score
        )

        print(
            f"k={k} | "
            f"Silhouette Score: {score:.4f}"
        )

    # ---------------------------------------------------------
    # Select k with the highest silhouette score
    # ---------------------------------------------------------

    selected_k = max(
        silhouette_scores,
        key=silhouette_scores.get,
    )

    selected_k = int(
        selected_k
    )

    # ---------------------------------------------------------
    # Train final clustering model
    # ---------------------------------------------------------

    final_model = KMeans(
        n_clusters=selected_k,
        random_state=random_seed,
        n_init=10,
    )

    cluster_labels = final_model.fit_predict(
        X_scaled
    )

    # ---------------------------------------------------------
    # Save clustering model and scaler
    # ---------------------------------------------------------

    clustering_model = {
        "model": final_model,
        "scaler": scaler,
    }

    model_path = (
        models_dir
        / "clustering_model.joblib"
    )

    joblib.dump(
        clustering_model,
        model_path,
    )

    # ---------------------------------------------------------
    # Save clusters.csv
    # ---------------------------------------------------------

    clusters_df = df[
        ["record_id"]
    ].copy()

    clusters_df["cluster"] = (
        cluster_labels
    )

    clusters_path = (
        output_dir
        / "clusters.csv"
    )

    clusters_df.to_csv(
        clusters_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Save clustering metrics
    # ---------------------------------------------------------

    metrics = {
        "method": "KMeans",
        "random_seed": random_seed,
        "features_used": FEATURE_NAMES,
        "k_values_tested": [
            2,
            3,
            4,
            5,
        ],
        "silhouette_scores": silhouette_scores,
        "selected_k": selected_k,
        "selected_silhouette_score": (
            silhouette_scores[
                str(selected_k)
            ]
        ),
        "interpretation_note": (
            "Clusters represent groups of records "
            "with similar input-feature profiles. "
            "They are not verified real-world categories."
        ),
    }

    metrics_path = (
        output_dir
        / "clustering_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # ---------------------------------------------------------
    # Create cluster visualization
    #
    # The plot uses the first two standardized features
    # only for visualization. Clustering itself used all
    # six features.
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    scatter = ax.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=cluster_labels,
        alpha=0.8,
    )

    ax.set_title(
        f"Collection Point Clusters (k={selected_k})"
    )

    ax.set_xlabel(
        "Standardized Plot Area (ha)"
    )

    ax.set_ylabel(
        "Standardized Rainfall (mm)"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.colorbar(
        scatter,
        ax=ax,
        label="Cluster",
    )

    fig.tight_layout()

    plot_path = (
        output_dir
        / "cluster_plot.png"
    )

    fig.savefig(
        plot_path,
        dpi=150,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    print("\nClustering completed.")

    print(
        f"Selected k: {selected_k}"
    )

    print(
        f"Selected silhouette score: "
        f"{silhouette_scores[str(selected_k)]:.4f}"
    )

    print("\nCluster counts:")

    print(
        pd.Series(
            cluster_labels
        )
        .value_counts()
        .sort_index()
    )

    print(
        f"\nMetrics saved to: {metrics_path}"
    )

    print(
        f"Clusters saved to: {clusters_path}"
    )

    print(
        f"Cluster plot saved to: {plot_path}"
    )

    print(
        f"Clustering model saved to: {model_path}"
    )