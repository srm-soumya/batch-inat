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

### Clone the sample git repository for this project
```sh
git clone https://github.com/srm-soumya/batch-inat
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

3. Upload your model, scripts and output to the azure file share.
You can find all the required files under the repository you cloned.(Make sure to delete the files inside the metadata folder, so that you can later create your own map files there)
Go to portal.azure.com, under your storage account check for Access keys. There will be 2 keys, use any of it.
```sh
azcopy --source scripts --destination https://<storage-name>.file.core.windows.net/<share-name>/<dir-name>/scripts --dest-key <destination-key> --recursive
azcopy --source model --destination https://<storage-name>.file.core.windows.net/<share-name>/<dir-name>/model --dest-key <destination-key> --recursive
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

### Check your cluster status
```sh
az batchai cluster list -o table
```
Wait till your cluster state is ```steady``` and all your nodes are in ```idle``` state.

### Create the jobs to run in your cluster

We need to run two jobs in our cluster.
- First job to create the required map files and store it in our ```metadata``` directory.
- Second job to run our model in the clusters. It will train the model and store it in the ```model``` directory and it will store all the outputs in the ```output``` directory.

#### First job

1. Create a ```job1.json``` file for creating the first job.
```json
{
    "properties": {
        "nodeCount": 1,
        "cntkSettings": {
            "pythonScriptFilePath": "$AZ_BATCHAI_INPUT_SCRIPT/02_model.py",
            "commandLineArgs": "--preprocess -d $AZ_BATCHAI_INPUT_DATA -dd $AZ_BATCHAI_OUTPUT_OUT",
            "process_count": 1
        },
        "stdOutErrPathPrefix": "$AZ_BATCHAI_MOUNT_ROOT/afs",
        "inputDirectories": [
            {
                "id": "SCRIPT",
                "path": "$AZ_BATCHAI_MOUNT_ROOT/afs/inatdir/scripts"
            },
            {
                "id": "MODEL",
                "path": "$AZ_BATCHAI_MOUNT_ROOT/afs/inatdir/model"
            },
            {
                "id": "DATA",
                "path": "$AZ_BATCHAI_MOUNT_ROOT/nfs"
            }
        ],
        "outputDirectories": [
            {
                "id": "OUT",
                "pathPrefix": "$AZ_BATCHAI_MOUNT_ROOT/afs/inatdir/output",
                "pathSuffix": "Metadata"
            }
        ],
        "containerSettings": {
            "imageSourceRegistry": {
                "image": "microsoft/cntk:2.1-gpu-python3.5-cuda8.0-cudnn6.0"
            }
        }
    }
}
```

2. Create a job with <job-name> and run it on the cluster <cluster-name>.
```sh
az batchai job create -l eastus -g <group-name> -n <job-name> -r <cluster-name> -c job1.json
```


3. Monitor the job <job-name> to get an overview of the job status.
```sh
az batchai job list -o table
```

The ```executionState``` contains the current execution state of the job:
    - queued: the job is waiting for the cluster nodes to become available
    - running: the job is running
    - succeeded (or failed) : the job is completed and executionInfo contains details about the result

4. When your <job-name> status is in ```succeeded``` state get the required ```map_files``` from your account in portal.azure.com which will be under ```$AZ_BATCHAI_MOUNT_ROOT/afs/inatdir/output/subscription-id/<group-name>/jobs/<job-name>/job-id/outputs/Metadata```

Download the files and put it inside the ```metadata``` directory in your project folder.

5. Upload the metadata repository to the azure file share.
```sh
azcopy --source metadata --destination https://<storage-name>.file.core.windows.net/<share-name>/<dir-name>/metadata --dest-key <destination-key> --recursive
```

6. Delete the job <job-name> once you are done with it.
```sh
az batchai job delete -n <job-name>
```

#### Second job

1. Create a ```job2.json``` file for creating the second job.
```json
{
    "properties": {
        "nodeCount": 2,
        "cntkSettings": {
            "pythonScriptFilePath": "$AZ_BATCHAI_INPUT_SCRIPT/02_model.py",
            "commandLineArgs": "--train -d $AZ_BATCHAI_INPUT_DATA -dd $AZ_BATCHAI_INPUT_METADATA -m $AZ_BATCHAI_INPUT_MODEL -o $AZ_BATCHAI_OUTPUT_OUT",
            "processCount": 4
        },
        "stdOutErrPathPrefix": "$AZ_BATCHAI_MOUNT_ROOT/afs",
        "inputDirectories": [
            {
                "id": "DATA",
                "path": "$AZ_BATCHAI_MOUNT_ROOT/nfs"
            },
            {
                "id": "METADATA",
                "path": "$AZ_BATCHAI_MOUNT_ROOT/afs/inatdir/metadata"
            },
            {
                "id": "SCRIPT",
                "path": "$AZ_BATCHAI_MOUNT_ROOT/afs/inatdir/scripts"
            },
            {
                "id": "MODEL",
                "path": "$AZ_BATCHAI_MOUNT_ROOT/afs/inatdir/model"
            }
        ],
        "outputDirectories": [
            {
                "id": "OUT",
                "pathPrefix": "$AZ_BATCHAI_MOUNT_ROOT/afs/inatdir/output",
                "pathSuffix": "Models"
            }
        ],
        "containerSettings": {
            "imageSourceRegistry": {
                "image": "microsoft/cntk:2.1-gpu-python3.5-cuda8.0-cudnn6.0"
            }
        }
    }
}
```

2. Create another job with <job-name> and run it on the cluster <cluster-name>
```sh
az batchai job create -l eastus -g <group-name> -n <job-name> -r <cluster-name> -c job2.json
```

3. Monitor the job <job-name> to get an overview of the job status.
```sh
az batchai job list -o table
```

4. Check the status of your nodes in the cluster.
```sh
az batchai cluster list-nodes -n <cluster-name> -o table
```

This will list the cluster details like IP and port number, you can ssh into them and monitor the GPU usage and perform any other task.


5. List stdout and stderr files
You can look into the ```stdout``` and ```stderr``` log files. To list the links to these files use:
```sh
az batchai job list-files --name <job-name> --output-directory-id stdouterr
```
This will give you the name & url of your log files.

You can then stream the ```stderr``` or ```stdout``` log files using:
```
az batchai job stream-file --job-name <job-name> --output-directory-id stdouterr --name stderr.txt
az batchai job stream-file --job-name <job-name> --output-directory-id stdouterr --name stdout.txt
```

6. When your <job-name> status is in ```succeeded``` state, you can get your ```trained_model```  and ```log_files``` from your training under ```$AZ_BATCHAI_MOUNT_ROOT/afs/inatdir/output/subscription-id/<group-name>/jobs/<job-name>/job-id/outputs/Models```


### Delete resources
Once you are done with your training, make sure to delete the jobs and clusters.

1. Delete the jobs
```sh
az batchai job delete -n <job-name>
```

2. Delete the clusters
```sh
az batchai cluster delete -n <cluster-name>
```



