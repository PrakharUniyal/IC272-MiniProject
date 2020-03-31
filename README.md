# Description of Dataset and Problem Statement

This dataset contains data of performance of a broadband gateway network (BNG) device. BNG device is the access point for subscribers, through which they connect to the broadband network. When a connection is established between BNG device and CustomerPremise Equipment (CPE), the subscriber can access the broadband services provided by the Network Service Provider (NSP) or Internet Service Provider (ISP). A network management company monitors the performance of each of the BNG devices to provide better services. The company records different performance measures (PMs) for every 15 minutes. A description of different columns i.e.attributes (PMs) in this dataset is detailed below.

1. **CreationTime:** Date and time of the recording of sample

1. **AuthenticateCount:** Number of active subscribers authenticated their connection.  

1. **ActiveCount:** Number of active subscribers connected to the device.

1. **DisconnectCount:** Number of active subscribers disconnected from the device.

1. **CPUUtil:** contains the % of usage of processor in the device.

1. **MemoryUsed:** Total memory in Bytes used in the device

1. **MemoryFree:** Total of memory in Bytes free in the device

1. **TempMin:** Minimum temperatures among the temperatures recorded from the different slots in the device

1. **TempMax:** Maximum temperatures among the temperatures recorded from the different slots in the device

1. **TempAvg:** Average temperatures among the temperatures recorded from the different slots in the device

1. **InBandwidth:** Total bandwidth utilization in Bytes from the input ports of all the interfaces. 

1. **OutBandwidth:** Total bandwidth utilization in Bytes from the output ports of all the interfaces.

1. **InTotalPPS:** Total packets per second transmitted from the input ports of all the interfaces.

1. **OutTotalPPS:** Total packets per second transmitted from the output ports of all the interfaces.

--------------------------------------------------------------------------------------------------------------------
70% of data has been considered for training and remaining 30% of data for testing.

* Data preprocessing has been performed.
* Descriptive analytics has been performed to understand the data and infer from the data. 
* Predictive analysis (regressive analysis) has been performed for predicting the **MemoryUsed** using different regression techniques.

Inferences have been provided in the final report.
