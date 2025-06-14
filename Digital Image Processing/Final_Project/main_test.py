import os
import glob
import argparse
import json
import matlab.engine
from PIL import Image
import numpy as np
import logging
import pytesseract
import Levenshtein
import matplotlib.pyplot as plt

# --- 設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 輔助函式 ---
def calculate_cer(ocr_text, ground_truth_text):
    """計算字元錯誤率 (Character Error Rate, CER)"""
    if not ground_truth_text: return float('inf')
    ocr_text = ocr_text.strip()
    distance = Levenshtein.distance(ocr_text, ground_truth_text)
    return distance / len(ground_truth_text)

def setup_matlab_engine():
    """啟動 MATLAB 引擎並設定好環境。"""
    logging.info("正在啟動 MATLAB 引擎...")
    try:
        eng = matlab.engine.start_matlab()
        logging.info("MATLAB 引擎已啟動。")
        eng.clear('functions', nargout=0)
        project_path = os.getcwd()
        eng.cd(project_path, nargout=0)
        eng.addpath(eng.genpath(project_path), nargout=0)
        return eng
    except Exception as e:
        logging.error(f"啟動 MATLAB 引擎失敗: {e}")
        return None

def create_comparison_image(original_img, intermediate_imgs, final_img, image_name, diagnosis_report, ocr_results, output_path):
    """使用 matplotlib 建立包含完整診斷與 OCR 資訊的 2x3 比較圖，並加入內部診斷日誌。"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Image Processing Analysis for: {image_name}', fontsize=20)
    axes = axes.flatten()

    titles = ['1. Original', '2. Grayscale', '3. After Lighting Stage', '4. After Noise Stage', '5. After Sharpening Stage', '6. Final Binary (OCR Input)']
    images_to_plot = [original_img] + intermediate_imgs + [final_img]

    logging.info("--- ENTERING PLOTTING FUNCTION ---")
    logging.info(f"準備繪製 {len(images_to_plot)} 張圖片到 {len(axes)} 個子圖中...")

    for i, ax in enumerate(axes):
        # 【新增診斷】: 在繪製前印出當前狀態
        logging.info(f"  - 準備繪製子圖 #{i+1}...")

        if i < len(images_to_plot):
            img = images_to_plot[i]
            title = titles[i]
            logging.info(f"    - Title: '{title}', Image Type: {type(img)}, Image Mode: {img.mode}, Image Size: {img.size}")
            
            cmap = 'gray' if i > 0 else None
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
            ax.set_title(title, fontsize=14, pad=20)
            ax.axis('off')

            # ... (此處標示 Applied/Skipped 和 OCR 結果的程式碼不變) ...
            status_text, status_color = '', 'gray'
            if i == 2: # After Lighting
                status = diagnosis_report.get('lighting', 'ok')
                if status not in ['ok']: status_text, status_color = f"Applied: {status}", 'lightgreen'
                else: status_text, status_color = "SKIPPED", 'lightgray'
            elif i == 3: # After Noise
                status = diagnosis_report.get('noise', 'none')
                if status not in ['none']: status_text, status_color = f"Applied: {status}", 'lightgreen'
                else: status_text, status_color = "SKIPPED", 'lightgray'
            elif i == 4: # After Sharpening
                status = diagnosis_report.get('blur', False)
                if status: status_text, status_color = "Applied", 'lightgreen'
                else: status_text, status_color = "SKIPPED", 'lightgray'
            if status_text:
                ax.text(0.05, 0.95, status_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor=status_color, alpha=0.8))
            ocr_text_to_show = ""
            if i == 0 and ocr_results: # Original
                ocr_before = ocr_results['before'][:30] + "..." if len(ocr_results['before']) > 30 else ocr_results['before']
                ocr_text_to_show = f"OCR Before: '{ocr_before}'\nCER: {ocr_results['cer_before']:.2%}"
            elif i == 5 and ocr_results: # Final
                truth = ocr_results['truth'][:25] + "..." if len(ocr_results['truth']) > 25 else ocr_results['truth']
                ocr_after = ocr_results['after'][:25] + "..." if len(ocr_results['after']) > 25 else ocr_results['after']
                ocr_text_to_show = (f"Ground Truth: '{truth}'\n" f"OCR After: '{ocr_after}'\n" f"CER: {ocr_results['cer_after']:.2%}")
            if ocr_text_to_show:
                ax.text(0.5, -0.15, ocr_text_to_show, size=10, ha="center", transform=ax.transAxes, fontfamily='monospace', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        else:
            logging.info(f"    - 子圖 #{i+1} 無對應圖片，將其隱藏。")
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"資訊比較圖已儲存至: {output_path}")   
def process_single_image(eng, image_path, output_dir, ground_truth_data):
    """對單一影像執行處理，並加入最終的詳細診斷訊息。"""
    image_name = os.path.basename(image_path)
    logging.info(f"--- 開始處理影像: {image_name} ---")

    try:
        final_matrix, intermediate_stages_list, report_struct = eng.functions.process_image(image_path, nargout=3)
        
        # 取得列表長度
        num_stages_received = len(intermediate_stages_list)
        logging.info(f"從 MATLAB 收到了 {num_stages_received} 個中間階段影像。 (預期為 4)")

        original_image = Image.open(image_path)
        width, height = original_image.convert('L').size
        correct_shape = (height, width)
        
        # 轉換中間影像
        intermediate_pil_images = []
        for i, stage_matrix in enumerate(intermediate_stages_list):
            numpy_array = np.array(stage_matrix._data, dtype=np.uint8).reshape(correct_shape, order='F')
            intermediate_pil_images.append(Image.fromarray(numpy_array))
        
        # 轉換最終影像
        final_numpy = np.array(final_matrix._data, dtype=np.uint8).reshape(correct_shape, order='F') * 255
        processed_image = Image.fromarray(final_numpy)
        
        diagnosis_report = {k: report_struct[k] for k in report_struct}
        
        # OCR 評估
        ocr_results = {}
        if image_name in ground_truth_data:
            ground_truth = ground_truth_data[image_name]['ground_truth_text']
            ocr_config = r'--psm 7'
            ocr_before = pytesseract.image_to_string(original_image, lang='eng+rus', config=ocr_config).strip()
            ocr_after = pytesseract.image_to_string(processed_image, lang='eng+rus', config=ocr_config).strip()
            cer_before = calculate_cer(ocr_before, ground_truth)
            cer_after = calculate_cer(ocr_after, ground_truth)
            ocr_results = {
                'before': ocr_before, 'after': ocr_after, 'truth': ground_truth,
                'cer_before': cer_before, 'cer_after': cer_after
            }
            logging.info(f"評估完成: CER Before: {cer_before:.2%} -> CER After: {cer_after:.2%}")
        else:
            logging.info("此影像無標準答案，跳過量化評估。")

        # --- 【X光檢查】: 在呼叫繪圖函式前，印出所有準備好的資料的詳細資訊 ---
        logging.info("--- PLOTTING PRE-CHECK (X-RAY) ---")
        logging.info(f"1. original_image: Type={type(original_image)}, Mode={original_image.mode}, Size={original_image.size}")
        logging.info(f"2. intermediate_pil_images: 數量={len(intermediate_pil_images)}")
        for i, img in enumerate(intermediate_pil_images):
            logging.info(f"   - Intermediate Img #{i+1}: Type={type(img)}, Mode={img.mode}, Size={img.size}")
        logging.info(f"3. processed_image (final_img): Type={type(processed_image)}, Mode={processed_image.mode}, Size={processed_image.size}")
        images_to_plot_for_check = [original_image] + intermediate_pil_images + [processed_image]
        logging.info(f"==> 最終要繪製的圖片總數為: {len(images_to_plot_for_check)} (預期為 6)")
        logging.info("------------------------------------")
        
        # 建立比較圖
        comp_image_path = os.path.join(output_dir, f"comparison_{os.path.splitext(image_name)[0]}.png")
        create_comparison_image(original_image, intermediate_pil_images, processed_image, image_name, diagnosis_report, ocr_results, comp_image_path)

    except Exception as e:
        lasterr = eng.evalc('lasterr')
        logging.error(f"處理影像 {image_name} 時發生 Python 錯誤: {e}")
        logging.error(f"詳細的 MATLAB 錯誤訊息: \n{lasterr}")
def main(args):
    """主函式，載入資料並驅動處理流程。"""
    input_dir, output_dir = args.input_dir, args.output_dir
    if not os.path.isdir(input_dir):
        logging.error(f"輸入路徑 '{input_dir}' 不存在或不是一個資料夾。")
        return
    os.makedirs(output_dir, exist_ok=True)
    ground_truth_data = {}
    json_path = os.path.join(input_dir, 'info.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f: ground_truth_data = json.load(f)
    else:
        logging.warning("未找到 'info.json'。")
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif'):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    logging.info(f"找到 {len(image_paths)} 張圖片，準備開始處理。")
    eng = setup_matlab_engine()
    if not eng: return
    try:
        for image_path in image_paths:
            process_single_image(eng, image_path, output_dir, ground_truth_data)
    finally:
        logging.info("所有任務完成，正在關閉 MATLAB 引擎...")
        eng.quit()
        logging.info("引擎已關閉。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 MATLAB 核心批次處理影像，並產生包含完整資訊的視覺化比較圖。")
    parser.add_argument("--input_dir", type=str, required=True, help="包含要處理影像與 info.json 的資料夾路徑。")
    parser.add_argument("--output_dir", type=str, default="output_results", help="儲存處理後影像與比較圖的資料夾路徑。")
    args = parser.parse_args()
    main(args)