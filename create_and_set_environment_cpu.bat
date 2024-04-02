call conda create -n datesnet python=3.9
call conda activate datesnet
set currDir=%cd%
set pipReqFile=%currDir%\requirements\pip_requirements.txt
call conda install --file %conReqFile%
call pip install -r %pipReqFile%
call pip install torch torchvision torchaudio