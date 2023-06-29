class VideoUploads:
    def __init__(self):
        self.filenames = []

    def Button_um_Video_hochzuladen(self):
        import ipywidgets as widgets
        import asyncio
        import os
        upload_button = widgets.FileUpload(description='Video Upload', accept='.mov', multiple=False)
        display(upload_button)
        def handle_upload_button_change(change):
            uploaded_files = upload_button.value
            filenames=[]
            for uploaded_file in uploaded_files:
                if uploaded_file:
                    file_name = uploaded_file['name']
                    file_contents = uploaded_file['content']
                    file_path = os.path.join(os.getcwd(), file_name)
                    with open(file_path, 'wb') as f:
                        f.write(file_contents)
                    filenames.append(file_name)
            self.filenames = filenames.copy()

            # Dateiname des hochgeladenen Videos in Datei schreiben, damit der Dateiname später ausgelesen werden kann.
            file = open('DateinameVideo.txt','w')
            file.write(file_name)
            file.close()
       
        upload_button.observe(handle_upload_button_change, names='value')
        
    def get_filenames(self):
        return self.filenames
