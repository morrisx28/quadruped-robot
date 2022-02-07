import os,sys
import pickle
import socket
import time
import getpass
import copy
from LearningUtility import TrainingConfiguration

class TrainigData():
    #Send by Server received by monitor
    def __init__(self):
        #Static Info
        self.learning_info     = (None, None) #len of state and len of action
        self.foot_name         = None
        self.gait_len          = 0 # gait.numberOfPoint
        self.each_target_times = 0
        self.action_range = [0, 0]
        #Dynamic Info
        self.motor_info        = [None, None, None]
        self.tsne_info         = (None, None) #Transition, Random flag ; Transition = (current_state, action, new_state, reward)
        self.dt                = None
        self.target            = [False, [0 ,0, False]]
        self.learning_duration = None
        self.loss              = 0
        self.reward_factor     = None
        self.each_reward       = None
        self.under_learning    = True
        self.real_time         = False
        self.loading           = [0, 0]
        self.explorer_rate     = 0
        self.ground_info       = None
        self.obstacle_info     = None
        self.obstacle_hit      = False
        self.target_passed     = 0
        self.real_motor_info   = [None, None, None]



class StatusCmd():
    #Send by Monitor received by server
    def __init__(self):
        self.check_learning = False
        self.Terminate      = False
        self.NeedSave       = [False , ""]
        self.manual_target  = [False, [0, -30]]
        self.real_time      = False
        self.under_learning = True
        self.reward_factor  = None
        self.state_monitor = [False , 1] # [on/off , Hz]


class StatusInfo(object):
    
    def __init__(self):
        
        self.status_cmd = StatusCmd()
        self.data_stru = TrainigData()
        self.NET_address = None
        self.NET_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.NET_sock.setblocking(0)
        self.data_sent_count = 0
        self.dot_count = 0
        user = getpass.getuser()
        if user in ['rudy3742', 'ericubuntu']:
            self.default_port = 51535
        elif user == 'clement':
            self.default_port = 58000
            if socket.gethostname() == 'lab931':
                self.default_port = 58205
        elif user == "ubuntu":
            self.default_port = 52000
        else:
            self.default_port = 50000

        # attr_dic = dir(self.status_cmd)
        # self.usefull_attr = [e for e in attr_dic if not e.startswith("__")]

    def UpdateSendObjectUDP(self):
        '''Send Training data to Monitor'''
        try:
            self.status_sock.sendto(pickle.dumps(self.data_stru),self.NET_address)
        except Exception as e:
            print("send error")
            print(e)
            print(len(pickle.dumps(self.data_stru)))
        self.data_sent_count += 1
        if self.status_cmd.check_learning:
            if self.status_cmd.state_monitor[0]:
                time.sleep(1/self.status_cmd.state_monitor[1])
            else:
                time.sleep(self.data_stru.dt/2)

    def SendConfigUDP(self, config):
        '''Send learning config to monitor'''
        try:
            self.status_sock.sendto(pickle.dumps(config),self.NET_address)
        except Exception as e:
            print("\nsend config error\n")
            print(e)
            print(len(pickle.dumps(config)))

    def StatusServerCreate(self):
        self.status_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

        if socket.gethostname() == 'Server2':
            address = ('192.168.1.101', self.default_port)
            print("port" + str(self.default_port))
        elif socket.gethostname() == 'ERIC-PERSONAL':
            address = ('192.168.100.108',self.default_port)
        elif socket.gethostname() == 'Erics-MacBook-Pro-13.local':
            address = ('localhost',self.default_port)
        elif socket.gethostname() == 'nvidia-jetson-nano':
            address = ('192.168.1.19', self.default_port)
        elif socket.gethostname() == 'ericubuntu':
            address = ('192.168.100.108', self.default_port)
        elif socket.gethostname() == 'lab931':
            address = ('192.168.1.243', self.default_port)
        elif socket.gethostname() == 'vm1596442227804-1110436-iaas':
            address = ("192.168.211.22", self.default_port)
        elif socket.gethostname() == 'vm1596442868367-1110475-iaas':
            address = ("192.168.211.31", self.default_port)
        elif socket.gethostname() == 'vm1596467797843-1111219-iaas':
            address = ("192.168.211.19", self.default_port)
        elif socket.gethostname().endswith("iaas"):
            host_name = socket.gethostname()
            address = (socket.gethostbyname(host_name), self.default_port)
        else:
            address = ('localhost',self.default_port)

        try:
            self.status_sock.bind(address)
            self.status_sock.setblocking(0)
        except OSError as e:
            self.default_port += 1
            print("Error Occur as {0}, change port to {1}".format(e,self.default_port))
            self.StatusServerCreate()
        


    def StatusReceive(self):
        try:
            cmd_received, self.NET_address = self.status_sock.recvfrom(512)
            cmd_received = pickle.loads(cmd_received)
            self.status_cmd = cmd_received

        except socket.error as _:
            pass

    def StatusSend(self, ServerName, port=58000):
        # with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
        if ServerName == 'Server2':
            address = ('nchulab931.ddns.net',port)
        elif ServerName == 'ERIC_PERSONAL':
            # address = ('192.168.100.108',port)
            address = ('ejt3742dev.ddns.net',port)
        elif ServerName == 'Nano':
            address = ('192.168.1.19', port)
        elif ServerName == 'localhost':
            address = ('localhost' ,port)
        elif ServerName == 'Ubuntu1060':
            address = ('nchulab931.ddns.net', port)
        elif ServerName == "twcc1":
            address = ("103.124.73.1", port)
        elif ServerName == "twcc2":
            address = ("103.124.73.116", port)
        elif ServerName == "twcc3":
            address = ("103.124.73.150", port)
        elif "." in ServerName:
            address = (ServerName, port)
        self.NET_sock.sendto(pickle.dumps(self.status_cmd),address)

    def printSocketMsg(self):
        # if self.data_sent_count % 25 == 0:
        x = "." * ((self.dot_count%4))
        while len(x) < 3:
            x = x + " "
        print("sending data to : " + str(self.NET_address) + " through NET service" + x, end = '\r')
        self.dot_count += 1

    def resetSocket(self):
        self.NET_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

#Following is uesd for real-world learning

class RealWorldLearningTransition():
    #UDP package
    def __init__(self):
        self.package_type = 'transition'
        self.current_state = None
        self.action  = 0
        self.new_info  = None
        self.new_state = None
        self.target  = None
        self.explore_rate = 0.
        self.random_flag = False
        self.gait_index = 0
        self.time = None
        self.index = 0
        self.dummy_motor_info = [None, None, None]

class ServerRequest():
    #TCP package
    def __init__(self):
        self.package_type = 'request'
        self.request_for_new_info  = False
        self.request_for_new_model = False
        self.real_model_terminated = False
        self.memory_over_sized     = False

BUFF_SIZE = 64*1024

class RealworldTrainingClient(object):
    
    def __init__(self):
        self.learning_sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.request = ServerRequest()
        self.sent_count = 0
        self.transition_count = 1

        # self.transition_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.transition = RealWorldLearningTransition()

    def connectServer(self, server_address, port=52535):
        self.server_address = server_address
        self.port = port
        try:
            self.learning_sock.connect((self.server_address,self.port))
        except Exception as e:
            print(e)
            print("Connect Failed!!!")
            sys.exit(0)
        self.sentRequest(new_info=True, new_model=True)
        lr_info = self.getLearningInfo()
        # lr_info = self.receiveTCP_Data()
        return lr_info

    def sentData(self, data_obj):
        data = pickle.dumps(data_obj)
        data_len = len(data)
        try:
            data = bytes(f"{data_len:4d}", encoding="utf8") + data
            # print(f"{self.sent_count} : {data}")
            self.learning_sock.sendall(data)
            self.sent_count += 1
        except Exception as e:
            print(f"send error after {self.sent_count}")
            print(e)
            print(len(pickle.dumps(self.transition)))
            sys.exit(0)

    def sentTransition(self):
        # try:
        #     # self.socket.sendto(pickle.dumps(self.data_stru),self.NET_address)
        #     # self.transition_sock.sendto(
        #     #     pickle.dumps(self.transition), (self.server_address, self.port+1)
        #     #     )
        #     data = pickle.dumps(self.transition)
        #     data_len = len(data)
        #     self.learning_sock.send(bytes(f"{data_len:4d}", encoding="utf8") + data)
        # except Exception as e:
        #     print("send error")
        #     print(e)
        #     print(len(pickle.dumps(self.transition)))
        #     sys.exit(0)
        self.transition.index = self.transition_count
        self.sentData(self.transition)
        self.transition_count += 1

    def sentRequest(self, new_info=False, new_model=False):
        self.request = ServerRequest()
        self.request.request_for_new_info = new_info
        self.request.request_for_new_model = new_model
        self.sentData(self.request)
        # data = pickle.dumps(self.request)
        # data_len = len(data)
        # self.learning_sock.send(bytes(f"{data_len:4d}", encoding="utf8") + data)
        # print(data_len)
        

    def sentEndSignal(self):
        self.request = ServerRequest()
        self.request.real_model_terminated = True
        # data = pickle.dumps
        # self.learning_sock.send(pickle.dumps(self.request))
        self.sentData(self.request)

    def sentMemoryLeakSignal(self):
        self.request = ServerRequest()
        self.request.memory_over_sized = True
        self.sentData(self.request)
        # self.learning_sock.send(pickle.dumps(self.request))

    def getLearningInfo(self):
        lr_info = self.receiveTCP_Data()
        return lr_info if lr_info is not None else LearningInfo()

    def receiveTCP_Data(self):
        data = b''
        # data = []
        #Reciving pre-Info for the model data
        recv_info = pickle.loads(self.learning_sock.recv(4096))

        print("    recv_info received, sending go flag")
        go_flag = True
        self.learning_sock.send(pickle.dumps(go_flag))
        time.sleep(1)
        count = 0
        data_len = 0
        # while True:
        rv_error = False
        rv_time = time.time()
        while data_len != recv_info[1]:
            part = self.learning_sock.recv(BUFF_SIZE)
            data += part
            count += 1
            data_len = sys.getsizeof(data)
            print("    Received {0:.1f} percent of data".format(data_len/recv_info[1]*100), end = '\r')
            if time.time() - rv_time >= 5*60*60:
                rv_error = True
                break
            # print("    Received {0:.1f} percent of data".format(count/recv_info[0]*100), end = '\r')
            # if len(part) < BUFF_SIZE:
            #     break
        if not rv_error:
            print("Data received")
            # for i in range(recv_info[0]+1):
            #     part = self.learning_sock.recv(BUFF_SIZE)
            #     # if not part:
            #     #     print("No data left")
            #     #     print("length of data is {0}".format(4*len(data)))
            #     #     break
            #     if part != b'':
            #         data.append(part)
            #         print("data {0} , len {1}".format(len(data), sys.getsizeof(data[i])))
            #     # data += part
            #     # if len(part) < BUFF_SIZE:
            #     #     print("data end")
            #     #     data_end = True
            #         # either 0 or end of data
            #         # break
            print("reassemble {0} bytes of data".format(sys.getsizeof(data)))
            # info = pickle.loads(b"".join(data))
            # info = pickle.loads(data_arr)
            info = pickle.loads(data)
            return info
        else:
            print("Data received Error")
            return None

    def closeSock(self):
        self.learning_sock.close()
        # self.transition_sock.close()
        print("Client Socket Closed!")

class LearningInfo():
    #Package sent to node
    def __init__(self):
        self.learning_conf      = None
        self.model_state_dirc   = None
        self.TerminateLearning  = False

class RealworldTrainingServer(object):

    def __init__(self, address='localhost', port=52535):
        #TCP for learning instruction
        self.learning_sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.learning_sock.bind((address,port))
        self.learning_sock.listen(1)
        self.learning_sock.setblocking(True)

        #UDP for transition
        # self.transition_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        # self.transition_sock.bind((address, port+1))
        # self.transition_sock.setblocking(False)

        self.learning_address = None
        self.learning_connection = None
        self.scan_client_interval = 20 #seconds
        self.last_connection_time = None
        self.receive_count        = 0
        self.sent_time_out        = 10 #seconds
        self.last_data            = b''

        self.lr_info = LearningInfo()

    def waitforConnect(self, lr_conf=None):
        print("Wait for training node connection")
        self.learning_sock.setblocking(True)
        self.learning_connection, self.learning_address = self.learning_sock.accept()
        self.learning_sock.setblocking(False)
        self.learning_connection.setblocking(False)
        self.last_connection_time = time.time()
        print("learning node connected")
        # self.lr_info.learning_conf = lr_conf
        # self.lr_info.model_state_dirc = model_state_dirc
        # self.learning_connection.send(pickle.dumps(self.lr_info))
        # print("Model send")
        # self.lr_info = LearningInfo()   #reinitial package 

    def renewConnection(self, lr_conf=None):
        if time.time() - self.last_connection_time > self.scan_client_interval:
            try:
                self.learning_connection, self.learning_address = self.learning_sock.accept()
                print("New client connected, old clinet dumped!")
            except socket.error as _:
                print("No incoming connection")
        else:
            pass

    def processDataFormClient(self):
        try:
            r_data = b''
            r_data = r_data + self.learning_connection.recv(64)
            data_len = int(r_data[0:4])
            
            while len(r_data) != data_len+4:
                # print(data)
                r_data = r_data + self.learning_connection.recv(data_len+4-len(r_data))
            # print(f'{data_len} received: {len(data)}', end = '\r')
            # print("", end='\r')
            time.sleep(0.003)
            data = pickle.loads(r_data[4:])
            self.last_data = r_data
            self.receive_count += 1
            return data
        except socket.error as _:
            return None
        except Exception as e:
            print(e)
            print(len(r_data))
            print(self.last_data)
            print("")
            print(f"{self.receive_count}: {r_data}")
            print("")
            print(self.learning_connection.recv(512))
            sys.exit()

    # def processTransition(self):
    #     try:
    #         transition = pickle.loads(self.transition_sock.recv(4096))
    #         # transition = pickle.loads(self.learning_connection.recv(4096))
    #         return transition
    #     except socket.error as _:
    #         #empty transition
    #         return RealWorldLearningTransition()

    def getRequest(self):
        try:
            request = pickle.loads(self.learning_connection.recv(4096))
            print("Request received")
            return request
        except:
            #empty request
            return None

    def sentInfo(self):
        data = pickle.dumps(self.lr_info)
        data_len = sys.getsizeof(data)
        recv_info = [int(data_len/BUFF_SIZE), data_len]
        
        #Sent recv infomation to client
        print("Sending data length, d_lan: {0}, p_len: {1}".format(recv_info[1],recv_info[0]))
        self.learning_connection.send(pickle.dumps(recv_info))

        #Wait for go flag
        go_flag = False
        print("Waiting go flag for sending")

        client_has_no_response = False
        o_time = time.time()
        while not go_flag:
            try:
                go_flag = pickle.loads(self.learning_connection.recv(1024))
            except:
                pass
            if time.time() - o_time > self.sent_time_out:
                client_has_no_response = True
                break

        if client_has_no_response:
            print("client_has_no_response")
        else:
            print("Sending {0} bytes of data".format(data_len))
            self.learning_connection.setblocking(1)
            self.learning_connection.send(data)
            self.learning_connection.setblocking(0)
            print("data sent")
    # def sentModel(self, state_dirc):
    #     self.lr_info.model_state_dirc = state_dirc
    #     self.learning_connection.send(pickle.dumps(self.lr_info))
    #     self.lr_info.model_state_dirc = None

    def sentTerminate(self):
        self.lr_info.TerminateLearning = True
        self.learning_connection.send(pickle.dumps(self.lr_info))
        print("wait for terminate responce from node...")
        while not self.getRequest().real_model_terminated: pass
        print("Node terminated")

    def closeSock(self):
        self.learning_sock.close()
        # self.transition_sock.close()
        print("Server Socket Closed!")