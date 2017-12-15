Make sure you have an Azure Account, if not you can signup for a free subscription. [https://azure.microsoft.com/en-us/free/?WT.mc_id=A261C142F]

### Install azure-cli [https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest]

### Register Batch-AI resources for your account, it might take upto 15 minutes for registration
```sh
az provider register -n Microsoft.BatchAI
az provider register -n Microsoft.Batch
```

### Create a resource group
```sh
az group create --name <group-name> --location eastus
```

### Create an azure storage account
We will be using this storage account to host our model, metadata, scripts and output.
```sh
az storage account create -n <storage-name> --sku Standard_LRS -l eastus -g <group-name>
```

### Prepare azure file share (afs)

1. Create an azure file share.
We will use this azure file share (afs) to host our model, metadata, scripts and output.
```sh
az storage share create --name <share-name> --account-name <storage-name>
```

2. Create a directory inside your azure file share.
```sh
az storage directory create --name <dir-name> --share-name <share-name> --account-name <storage-name>
```

3. Upload your model, metadata, scripts and output to the azure file share.
You can get all this from the github repo []
Go to portal.azure.com, under your storage account check for Access keys. There will be 2 keys, use any of it.
```sh
azcopy --source scripts --destination https://<storage-name>.file.core.windows.net/<share-name>/<dir-name>/scripts --dest-key <destination-key> --recursive
azcopy --source model --destination https://<storage-name>.file.core.windows.net/<share-name>/<dir-name>/model --dest-key <destination-key> --recursive
azcopy --source metadata --destination https://<storage-name>.file.core.windows.net/<share-name>/<dir-name>/metadata --dest-key <destination-key> --recursive
azcopy --source output --destination https://<storage-name>.file.core.windows.net/<share-name>/<dir-name>/output --dest-key <destination-key> --recursive
```

### Prepare network file share (nfs)
1. Create a network file share
We will use nfs to host our training, validation and test images.
```sh
az batchai file-server create -n <nfs-name> -g <group-name> -l eastus -u <user-name> -p <password> --vm-size Standard_DS2_V2 --disk-count 1 --disk-size 500 --storage-sku Standard_LRS
```

2. ```ssh``` into network file server
This will give you ```Public IP``` to your server and you can ```ssh``` into it using the <user-name> & <password> you created in the previous step.
```sh
az batchai file-server list -o table
ssh <user-name>@Public-IP
```

3. Download and get the data inside ```/mnt``` directory in the network file server, you might have to use sudo access for this.
- Download your data
- untar
- rename it to data
- Logout from the system

```sh
cd /mnt
wget <path-to-data>
tar -zxvf <downloaded-tar-file>
mv <untar-file> data
exit
```

4. Copy ```scripts/01_structure_data.py``` from your system to ```/mnt/.``` in network file server.
```sh
scp scripts/01_structure_data.py <user-name>@Public-IP:~/
ssh <user-name>@Public-IP
cd /mnt
mv ~/01_structure_data.py .
```

5. Run the pre-processing script ```01_structure_data.py``` to split your data into train, validation set.
You might have to install the required python dependencies for this.
```sh
python 01_structure_data.py
```
Logout of the system

### Create a Cluster

- We need to create a cluster of VMs with UbuntuDSVM (Data Science Virtual Machine) Images, which come pre-installed with all the required libraries and tools.
- Number of Nodes and size of the machine are problem dependent.
- NC6 Machine => 1 K80 GPU (approx $1 an hour)
- NC12 Machine => 2 K80 GPU (approx $2 an hour)
- NC24 Machine => 4 K80 GPU (approx $4 an hour)
* The prices may vary based on regions in which you create your cluster.

You can play around with number of nodes and the size of machine you wish to create.
num of nodes * machine size => cost of your cluster.
for eg: If you create 2 NC12 Machines, it might cost you 2 * $2 => $4 an hour

While creating the cluster you can mount all your storage options like network file server, azure file share or blob storage.
for eg: To mount an azure file share at $AZ_BATCHAI_MOUNT_ROOT/afs ```--afs-name <share-name> --afs-mount-path afs```
$AZ_BATCHAI_MOUNT_ROOT is an environment variable that you can access.

We will mount the following storage files:
<nfs-name> $AZ_BATCHAI_MOUNT_ROOT/nfs
<share-name> $AZ_BATCHAI_MOUNT_ROOT/afs

We will create a 2 node NC12 UbuntuDSVM in this example. ```min``` and ```max``` arguments could be used to define the number of nodes you want to create in your cluster.
```
az batchai cluster create -l eastus -g <group-name> -n <cluster-name> --storage-account-name <storage-name> --nfs <nfs-name> --nfs-mount-path nfs --afs-name <share-name> --afs-mount-path afs -s Standard_NC12 -i UbuntuDSVM --min 2 --max 2 -u <user-name> -p <password>
```

#### Check your cluster status
```sh
az batchai cluster list -o table
```
Wait till your cluster state is ```steady``` and all your nodes are in ```idle``` state.

### Create the jobs to run in your cluster

We need to run two jobs in our cluster.
- First job to create the required map files and store it in our ```metadata``` directory.
- Second job to run our model in the clusters. It will train the model and store it in the ```model``` directory and it will store all the outputs in the ```output``` directory.

First job
```sh
az batchai job create -l eastus -g <group-name> -n <job-name> -r <cluster-name> -c job1.json
```

Second job
```sh
az batchai job create -l eastus -g <group-name> -n <job-name> -r <cluster-name> -c job2.json
```

You can find the job files in the github repo []



