import sys
import os
import numpy as np
from PIL import Image, ImageQt, ImageDraw, ImageFont
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QTextEdit,
    QDialog,
    QLineEdit,
    QMessageBox,
)
from PyQt6.QtGui import QPixmap, QIntValidator
from PyQt6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class EncoderDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("編碼畫面")
        self.setFixedSize(1000, 800)

        # 圖片預覽
        self.image_label = QLabel("尚未選擇影像")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.image_label.setFixedSize(350, 350)

        # 文字輸入
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("在此輸入要藏入的文字")
        self.text_edit.setFixedHeight(60)

        # 加密後圖片預覽
        self.out_image_label = QLabel("尚未加密影像")
        self.out_image_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.out_image_label.setFixedSize(350, 350)

        # 加密前圖片預覽下面的文字
        self.caption_before_label = QLabel("加密前影像")
        self.caption_before_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # 加密後圖片預覽下面的文字
        self.caption_after_label = QLabel("加密後影像")
        self.caption_after_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Matplotlib 畫布：左右兩個子圖
        self.fig = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax_before, self.ax_after = self.fig.subplots(1, 2)

        # 按鈕
        select_btn = QPushButton("選擇影像")
        select_btn.clicked.connect(self.select_image)
        exec_btn = QPushButton("執行並儲存")
        exec_btn.clicked.connect(self.update_and_save)

        # 建立第一組：原始圖 + 文字
        before_vbox = QVBoxLayout()
        before_vbox.addWidget(self.image_label)
        before_vbox.addWidget(self.caption_before_label)

        # 建立第二組：加密後圖 + 文字
        after_vbox = QVBoxLayout()
        after_vbox.addWidget(self.out_image_label)
        after_vbox.addWidget(self.caption_after_label)

        # 上半部佈局
        top_hbox = QHBoxLayout()
        btn_vbox = QVBoxLayout()
        btn_vbox.addWidget(select_btn)
        btn_vbox.addWidget(self.text_edit)
        btn_vbox.addWidget(exec_btn)
        btn_vbox.addStretch()

        top_hbox = QHBoxLayout()
        top_hbox.addLayout(btn_vbox)  # 按鈕區
        top_hbox.addLayout(before_vbox)  # 放原始圖＋文字
        top_hbox.addLayout(after_vbox)  # 放加密後圖＋文字

        # 主佈局
        main_vbox = QVBoxLayout()
        main_vbox.addLayout(top_hbox)
        main_vbox.addWidget(self.canvas)
        self.setLayout(main_vbox)

        # 暫存圖片路徑
        self.current_image_path = None

    # 取得要加密的文字，加密後前面是加密長度後面是加密內容
    # 例如：長度是 16 位元，內容是 8 位元的字元，則總共 24 位元
    def get_text_bits(self):
        text = self.text_edit.toPlainText()
        text_bytes = text.encode("utf-8")  # 用 UTF-8 編碼支援中文
        text_bits = ''.join(f"{byte:08b}" for byte in text_bytes)
        length_bits = f"{len(text_bits):016b}"  # 16 bits 表示長度
        return length_bits + text_bits


    # Histogram shift algorithm
    def embed_histogram_shift(self, arr, bits):
        """
        1. 計算 peak p
        2. 把所有 < p 的值都 -1（<p-1 的變成 <p-2，0的話直接丟掉）
        3. 在原本等於 p 的像素裡，依 bits 把需要的那些改成 p-1
        """
        # deep cpoy，避免改到原陣列
        arr_enc = arr.copy()
        # 1. 計算 peak p
        hist = np.bincount(arr_enc.flatten(), minlength=256)
        p = np.argmax(hist)

        # 2. 將所有 < p 的灰階值往左移 1
        #    注意先把所有 0 的位置標記，以後可以留意是否有丟掉
        mask_lt_p = arr_enc < p
        # 把小於 p 的都減 1
        arr_enc[mask_lt_p] = arr_enc[mask_lt_p] - 1

        # 3. 找出所有 原本 == p 的位置（因為 <p 的都已被移掉）
        idx_p = np.argwhere(arr == p)  # 注意用原陣列 arr 去定位 p
        if len(bits) > len(idx_p):
            raise ValueError("bits 太長，像素不夠藏")

        # 4. 依 bits 把 p → p-1（bit=1 才改，bit=0 保持 p）
        for bit, (i, j) in zip(bits, idx_p):
            if bit == "1":
                arr_enc[i, j] = p - 1

        return arr_enc, p

    # 選擇影像
    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇影像", "", "PNG Images (*.png)"
        )
        if not path:
            return
        self.current_image_path = path
        pil_img = Image.open(path).convert("L")

        qimg = ImageQt.ImageQt(pil_img)
        pix = QPixmap.fromImage(qimg).scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pix)


        # 4. 顯示
        self.image_label.setPixmap(pix)

    def update_and_save(self):
        if not self.current_image_path:
            dlg = QDialog(self)
            dlg.setWindowTitle("錯誤")
            lbl = QLabel("請先選擇影像！", dlg)
            QVBoxLayout(dlg).addWidget(lbl)
            dlg.exec()
            return

        # 讀灰階圖
        img = Image.open(self.current_image_path).convert("L")
        arr = np.array(img)
        bits = self.get_text_bits()
        arr_enc, p = self.embed_histogram_shift(arr, bits)

        # 顯示直方圖
        self.ax_before.clear()
        self.ax_before.hist(arr.flatten(), bins=256, range=(0, 255))
        self.ax_before.set_title(f"Before (peak={p})")
        self.ax_after.clear()
        self.ax_after.hist(arr_enc.flatten(), bins=256, range=(0, 255))
        self.ax_after.set_title(f"After (peak={p})")
        self.fig.tight_layout()
        self.canvas.draw()

        # 顯示加密後影像
        qimg = ImageQt.ImageQt(Image.fromarray(arr_enc))
        pix = QPixmap.fromImage(qimg).scaled(
            self.out_image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.out_image_label.setPixmap(pix)

        # 另存為
        save_dir = QFileDialog.getExistingDirectory(self, "選擇儲存資料夾")
        if save_dir:
            base = os.path.basename(self.current_image_path)
            name, ext = os.path.splitext(base)
            out_path = os.path.join(save_dir, f"{name}_enc.png")
            Image.fromarray(arr_enc).save(out_path, format="PNG")


class DecoderDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("解碼畫面")
        self.setFixedSize(1000, 800)

        # 圖片預覽
        self.image_label = QLabel("尚未選擇影像")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.image_label.setFixedSize(350, 350)

        # peak輸入
        self.number_input = QLineEdit()
        self.number_input.setPlaceholderText("請輸入peak值")
        # 設定只能輸入整數（0 到 9999 的範圍）
        int_validator = QIntValidator(0, 9999)
        self.number_input.setValidator(int_validator)
        
        # 加密後圖片預覽
        self.out_image_label = QLabel("尚未解密影像")
        self.out_image_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.out_image_label.setFixedSize(350, 350)

        # 加密前圖片預覽下面的文字
        self.caption_before_label = QLabel("解密前影像")
        self.caption_before_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # 加密後圖片預覽下面的文字
        self.caption_after_label = QLabel("解密後影像")
        self.caption_after_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Matplotlib 畫布：左右兩個子圖
        self.fig = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax_before, self.ax_after = self.fig.subplots(1, 2)

        # 按鈕
        select_btn = QPushButton("選擇影像")
        select_btn.clicked.connect(self.select_image)
        exec_btn = QPushButton("執行並儲存")
        exec_btn.clicked.connect(self.update_and_save)

        # 顯示解密後文字
        self.decrypted_text_label = QLabel("解密後文字：")
        self.decrypted_text_label.setWordWrap(True)
        self.decrypted_text_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.decrypted_text_label.setFixedHeight(100)

        # 建立第一組：原始圖 + 文字
        before_vbox = QVBoxLayout()
        before_vbox.addWidget(self.image_label)
        before_vbox.addWidget(self.caption_before_label)

        # 建立第二組：加密後圖 + 文字
        after_vbox = QVBoxLayout()
        after_vbox.addWidget(self.out_image_label)
        after_vbox.addWidget(self.caption_after_label)

        # 上半部佈局
        top_hbox = QHBoxLayout()
        btn_vbox = QVBoxLayout()
        btn_vbox.addWidget(select_btn)
        btn_vbox.addWidget(self.number_input)
        btn_vbox.addWidget(exec_btn)
        btn_vbox.addWidget(self.decrypted_text_label)  # 顯示解密後文字
        btn_vbox.addStretch()

        top_hbox = QHBoxLayout()
        top_hbox.addLayout(btn_vbox)  # 按鈕區
        top_hbox.addLayout(before_vbox)  # 放原始圖＋文字
        top_hbox.addLayout(after_vbox)  # 放加密後圖＋文字

        # 主佈局
        main_vbox = QVBoxLayout()
        main_vbox.addLayout(top_hbox)
        main_vbox.addWidget(self.canvas)
        self.setLayout(main_vbox)

        # 暫存圖片路徑
        self.current_image_path = None


    # Histogram decode algorithm
    def decode_histogram_shift(self, arr, peak):
        """
        從加密後的影像解出 bits 並還原原始影像
        1. 使用指定的 peak 值 p
        2. 掃描整張圖：
            - 如果是 p-1，表示 bit=1，改回 p
            - 如果是 p，表示 bit=0，不用改
        3. 所有 < p 的值 +1 還原（因為加密時 -1）
        """
        arr_dec = arr.copy()
        decode_bits = []
        p = peak

        # 解出 bits 並還原 p 區塊
        for i in range(arr_dec.shape[0]):  # 取出影像矩陣的row
            for j in range(arr_dec.shape[1]):  # 取出影像矩陣的column
                if arr_dec[i, j] == p - 1:
                    decode_bits.append("1")
                    arr_dec[i, j] = p
                elif arr_dec[i, j] == p:
                    decode_bits.append("0")

        # 還原p左邊的灰階值（<p的都加1）
        mask_lt_p = arr_dec < p
        arr_dec[mask_lt_p] += 1
        # print(decode_bits)
        return "".join(decode_bits), arr_dec

    # 選擇影像
    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇影像", "", "PNG Images (*.png)"
        )
        if not path:
            return
        self.current_image_path = path
        pil_img = Image.open(path).convert("L")

        qimg = ImageQt.ImageQt(pil_img)
        pix = QPixmap.fromImage(qimg).scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pix)

    # 更新並儲存
    def update_and_save(self):
        # 檢查是否已選擇影像
        if not self.current_image_path:
            QMessageBox.warning(self, "錯誤", "請先選擇影像！")
            return

        # 取得並驗證 peak 值
        peak_text = self.number_input.text()
        if not peak_text:
            QMessageBox.warning(self, "錯誤", "請先輸入有效的 peak 值！")
            return
        p = int(peak_text)

        # 讀取灰階影像並轉為 numpy 陣列
        img = Image.open(self.current_image_path).convert("L")
        arr = np.array(img)

        # 執行解碼演算法
        bits, arr_dec = self.decode_histogram_shift(arr, p)

        # 若總長度不足 16 bits，代表連記錄長度的資料都沒有，無法解碼
        if len(bits) < 16:
            QMessageBox.warning(self, "錯誤", "解碼失敗：資料不足")
            return
        # 取出前 16 bits，這是原本文字的位元長度
        length_bits = bits[:16]
        # 將長度從二進位轉為十進位，得到實際文字資料的總 bit 數
        data_length = int(length_bits, 2)
        # 根據資料長度，擷取出真正的文字資料位元（從第 17 位到第 16 + data_length 位）
        text_bits = bits[16:16+data_length]
        # 每 8 bits 組成 1 byte，還原 byte 陣列
        text_bytes = bytearray(int(text_bits[i:i+8], 2) for i in range(0, len(text_bits), 8))
        # 嘗試將 byte 陣列解碼為 UTF-8 文字
        try:
            text = text_bytes.decode("utf-8")  # 用 UTF-8 解碼還原文字
        except UnicodeDecodeError:
            text = "[解碼錯誤：不是有效的 UTF-8 編碼]"

        self.decrypted_text_label.setText(f"解密後文字：\n{text}")


        # 更新直方圖：左邊原圖，右邊解密後
        self.ax_before.clear()
        self.ax_before.hist(arr.flatten(), bins=256, range=(0, 255))
        self.ax_before.set_title("Before decoding")

        self.ax_after.clear()
        self.ax_after.hist(arr_dec.flatten(), bins=256, range=(0, 255))
        self.ax_after.set_title("After decoding")

        self.fig.tight_layout()
        self.canvas.draw()

        # 顯示解密後影像
        qimg_dec = ImageQt.ImageQt(Image.fromarray(arr_dec))
        pix_dec = QPixmap.fromImage(qimg_dec).scaled(
            self.out_image_label.width(),
            self.out_image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.out_image_label.setPixmap(pix_dec)



        # 另存還原後的影像
        save_dir = QFileDialog.getExistingDirectory(self, "選擇儲存資料夾")
        if save_dir:
            base = os.path.basename(self.current_image_path)
            name, ext = os.path.splitext(base)
            out_path = os.path.join(save_dir, f"{name}_dec.png")
            Image.fromarray(arr_dec).save(out_path, format="PNG")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("影像訊息編碼/解碼工具")
        self.setFixedSize(400, 200)

        encode_btn = QPushButton("開啟編碼畫面")
        encode_btn.clicked.connect(self.open_encoder)
        decode_btn = QPushButton("開啟解碼畫面")
        decode_btn.clicked.connect(self.open_decoder)

        hbox = QHBoxLayout()
        hbox.addWidget(encode_btn)
        hbox.addWidget(decode_btn)

        vbox = QVBoxLayout(self)
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addStretch(1)

    def open_encoder(self):
        dlg = EncoderDialog(self)
        dlg.exec()

    def open_decoder(self):
        dlg = DecoderDialog(self)
        dlg.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
