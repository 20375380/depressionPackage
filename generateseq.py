def generate_seq(self):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)",
                                               options=options)
    if file_name:
        print("Selected video file: ", file_name)
        video_path = file_name
        video_path = video_path.replace('/', '\\')
        print(video_path)
        file_name = os.path.basename(file_name)
        file_name = file_name.replace('.mp4', '.txt')
        file_excel = file_name.replace('.txt', '.xlsx')
        datasets_root = r'.\\picture'
        datasets_new_root = r'.\\fuse'
        txt8_path = os.path.join('.\\storage-8\\', file_name)
        excel = os.path.join('.\\excel\\', file_excel)
        seqGenerator = seq_generator(video_path, datasets_root, datasets_new_root, txt8_path, excel)  # 创建序列生成器
        seqGenerator.cut_frame()  # 裁剪视频
        seqGenerator.pretreatment()  # 融合图像
        seqGenerator.classify_main()  # 分类
        seqGenerator.last_predict()  # 分类
        picture_list = os.listdir(datasets_root)
        fuse_list = os.listdir(datasets_new_root)
        # 删除图片
        for picture_name in picture_list:
            picture_name = os.path.join(datasets_root, picture_name)  # 构造文件的完整路径
            os.remove(picture_name)  # 删除文件
        '''
        for fuse_name in fuse_list:
            fuse_name = os.path.join(datasets_new_root, fuse_name)
            os.remove(fuse_name)
        '''
