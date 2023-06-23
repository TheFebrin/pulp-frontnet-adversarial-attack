# pulp-frontnet-adversarial-attack

## Clone the pulp-frontnet repo
```
git clone https://github.com/idsia-robotics/pulp-frontnet.git
```
And rename to `pulp_frontnet`


## Download the data
Data will be kept in `/pulp_frontnet/PyTorch/Data/Data/`

```
$ cd PyTorch
$ curl https://drive.switch.ch/index.php/s/FMQOLsBlbLmZWxm/download -o pulp-frontnet-data.zip
$ unzip pulp-frontnet-data.zip
$ rm pulp-frontnet-data.zip
```

## Troubleshooting
* `ModuleNotFoundError: No module named 'nemo'`
    * `!pip install git+https://github.com/NVIDIA/NeMo.git --user`
    * or `!pip install nemo_toolkit --user`