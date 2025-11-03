"""Data Mining Mid-Project CLI"""
import click
from src.preprocess import preprocess_train, preprocess_test
from src.train import train_model
from src.train_poly import train_polynomial_model
from src.predict import predict
from src.registry import mark_best


@click.group()
def cli():
    """資料探勘期中專案 - 交通流量預測"""
    pass


@cli.command()
@click.option('--mode', type=click.Choice(['train', 'test']), required=True,
              help='處理模式：train（訓練集）或 test（測試集）')
@click.option('--drop-holiday/--keep-holiday', default=True,
              help='是否移除原始 holiday 欄位（預設：移除）')
def preprocess(mode, drop_holiday):
    """前處理資料（訓練集或測試集）

    訓練模式：產生 train_processed.csv、weather_categories.json、scaler.joblib
    測試模式：使用既有資源產生 test_processed.csv
    """
    if mode == 'train':
        preprocess_train(drop_holiday=drop_holiday)
    else:
        preprocess_test(drop_holiday=drop_holiday)


@cli.command()
@click.option('--full/--split', default=True,
              help='使用完整訓練集（--full）或分割驗證集（--split）（預設：--full）')
def train(full):
    """訓練 LinearRegression 模型並儲存"""
    click.echo("=== 訓練 LinearRegression 模型 ===")
    model_path = train_model(use_full_train=full)
    click.echo(f"\n✓ 完成！模型路徑: {model_path}")


@cli.command(name='train-poly')
@click.option('--degree', default=2, type=int,
              help='多項式次數（預設：2）')
@click.option('--interaction-only/--all-terms', default=False,
              help='是否只產生交互作用項（預設：包含所有項）')
@click.option('--feature-selection/--all-features', default=True,
              help='是否選擇重要特徵（預設：選擇）')
@click.option('--top-k', default=15, type=int,
              help='選擇前 k 個重要特徵（預設：15）')
@click.option('--full/--split', default=True,
              help='使用完整訓練集（--full）或分割驗證集（--split）（預設：--full）')
def train_poly(degree, interaction_only, feature_selection, top_k, full):
    """訓練 Polynomial Regression 模型並儲存"""
    click.echo("=== 訓練 Polynomial Regression 模型 ===")
    model_path = train_polynomial_model(
        degree=degree,
        interaction_only=interaction_only,
        use_feature_selection=feature_selection,
        top_k_features=top_k,
        use_full_train=full
    )
    click.echo(f"\n✓ 完成！模型路徑: {model_path}")


@cli.command(name='mark-best')
@click.option('--run', required=True,
              help='要標記為最佳的模型路徑（例如：blob/models/runs/20251102-123456-linear/model.joblib）')
def mark_best_cmd(run):
    """標記指定模型為最佳模型"""
    click.echo("=== 標記最佳模型 ===")
    mark_best(run_path=run)


@cli.command()
@click.option('--run', default='latest',
              help='使用的模型：latest（最新）、best（最佳）或具體路徑（預設：latest）')
def predict_cmd(run):
    """執行預測並產生提交檔"""
    click.echo(f"=== 預測（使用模型：{run}）===")
    submission_path = predict(run=run)
    click.echo(f"\n✓ 完成！提交檔: {submission_path}")


if __name__ == "__main__":
    cli()
