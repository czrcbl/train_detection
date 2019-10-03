from distutils.core import setup

setup(name='GGCNN Detection',
      version='0.1',
      description='Detect the grasps on image.',
      author='Cézar Lemos',
      author_email='cezarcbl@protonmail.com',
      url='',
      packages=['ggcnn_detection'],
      install_recquires=['torch', 'opencv-python']
     )
