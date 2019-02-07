# -*- coding: utf-8 -*-


#
# Created by: PyQt5 UI code generator 5.11.3
#


from PyQt5 import QtCore, QtGui, QtWidgets
import sqlite3
from configparser import ConfigParser, ExtendedInterpolation
import sys
import main4GUI as main
import numpy as np
sys.path.append('..')

class Ui_MainWindow(object):


    def loadData(self, select=1, fileName='test.ini'):
        if select == 1:
            fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Import configuration file", "","(*.ini)")

        if fileName:
            parser = ConfigParser(interpolation=ExtendedInterpolation())
            parser.read(fileName)
            sections = parser.sections()

            self.tableWidget.setRowCount(0)
            row = -1
            for section_num, section in enumerate(sections):
                items = parser.items(section)
                for item_num, item in enumerate(items):
                    row += 1
                    self.tableWidget.insertRow(row)
                    self.tableWidget.setItem(row,0, QtWidgets.QTableWidgetItem(item[0]))
                    self.tableWidget.item(row,0).setFlags(QtCore.Qt.ItemIsEditable)

                    self.tableWidget.setItem(row, 1, QtWidgets.QTableWidgetItem(item[1]))


    def saveData(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Save configuration file","","(*.ini)")
        if fileName:
            config = ConfigParser()
            config.add_section("simulation_parameters")
            for i in range(self.tableWidget.rowCount()):
                config.set("simulation_parameters",self.tableWidget.item(i,0).text(),self.tableWidget.item(i,1).text())
            with open(fileName, 'w') as f:
                config.write(f)

    def applySettings(self):
        pass

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(795, 578)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        self.runBtn = QtWidgets.QPushButton(self.centralwidget)
        self.runBtn.setObjectName("runBtn")
        self.verticalLayout.addWidget(self.runBtn)
        self.stopBtn = QtWidgets.QPushButton(self.centralwidget)
        self.stopBtn.setCheckable(False)
        self.stopBtn.setChecked(False)
        self.stopBtn.setObjectName("stopBtn")
        self.verticalLayout.addWidget(self.stopBtn)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_6.addWidget(self.label_4)
        self.checkBox_T1 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_T1.setChecked(True)
        self.checkBox_T1.setObjectName("checkBox_T1")
        self.verticalLayout_6.addWidget(self.checkBox_T1)
        self.checkBox_T2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_T2.setChecked(True)
        self.checkBox_T2.setObjectName("checkBox_T2")
        self.verticalLayout_6.addWidget(self.checkBox_T2)
        self.checkBox_I1 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_I1.setChecked(True)
        self.checkBox_I1.setObjectName("checkBox_I1")
        self.verticalLayout_6.addWidget(self.checkBox_I1)
        self.checkBox_I2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_I2.setChecked(True)
        self.checkBox_I2.setObjectName("checkBox_I2")
        self.verticalLayout_6.addWidget(self.checkBox_I2)
        self.checkBox_I3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_I3.setChecked(True)
        self.checkBox_I3.setObjectName("checkBox_I3")
        self.verticalLayout_6.addWidget(self.checkBox_I3)
        self.checkBox_I4 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_I4.setChecked(True)
        self.checkBox_I4.setObjectName("checkBox_I4")
        self.verticalLayout_6.addWidget(self.checkBox_I4)
        self.horizontalLayout_6.addLayout(self.verticalLayout_6)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem2)
        self.checkBox_createAnimation = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_createAnimation.setChecked(True)
        self.checkBox_createAnimation.setObjectName("checkBox_createAnimation")
        self.verticalLayout_5.addWidget(self.checkBox_createAnimation)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_7.addWidget(self.label_5)
        self.numIterationSample_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.numIterationSample_lineEdit.setObjectName("numIterationSample_lineEdit")
        self.horizontalLayout_7.addWidget(self.numIterationSample_lineEdit)
        self.verticalLayout_5.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_6.addLayout(self.verticalLayout_5)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_maps = QtWidgets.QVBoxLayout()
        self.verticalLayout_maps.setObjectName("verticalLayout_maps")
        self.Bathylabel = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Bathylabel.sizePolicy().hasHeightForWidth())
        self.Bathylabel.setSizePolicy(sizePolicy)
        self.Bathylabel.setObjectName("Bathylabel")
        self.verticalLayout_maps.addWidget(self.Bathylabel)
        self.rupertBtn = QtWidgets.QRadioButton(self.centralwidget)
        self.rupertBtn.setChecked(True)
        self.rupertBtn.setObjectName("rupertBtn")
        self.BathymetryBtnGrp = QtWidgets.QButtonGroup(MainWindow)
        self.BathymetryBtnGrp.setObjectName("BathymetryBtnGrp")
        self.BathymetryBtnGrp.addButton(self.rupertBtn)
        self.verticalLayout_maps.addWidget(self.rupertBtn)
        self.ShallowBtn = QtWidgets.QRadioButton(self.centralwidget)
        self.ShallowBtn.setObjectName("ShallowBtn")
        self.BathymetryBtnGrp.addButton(self.ShallowBtn)
        self.verticalLayout_maps.addWidget(self.ShallowBtn)
        self.SteeprBtn = QtWidgets.QRadioButton(self.centralwidget)
        self.SteeprBtn.setObjectName("SteeprBtn")
        self.BathymetryBtnGrp.addButton(self.SteeprBtn)
        self.verticalLayout_maps.addWidget(self.SteeprBtn)
        self.horizontalLayout_2.addLayout(self.verticalLayout_maps)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout_4.addWidget(self.label)
        self.numiterationsEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.numiterationsEdit.setObjectName("numiterationsEdit")
        self.horizontalLayout_4.addWidget(self.numiterationsEdit)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.velocitySallesBtn = QtWidgets.QRadioButton(self.centralwidget)
        self.velocitySallesBtn.setChecked(True)
        self.velocitySallesBtn.setObjectName("velocitySallesBtn")
        self.velocityBtnGrp = QtWidgets.QButtonGroup(MainWindow)
        self.velocityBtnGrp.setObjectName("velocityBtnGrp")
        self.velocityBtnGrp.addButton(self.velocitySallesBtn)
        self.verticalLayout_4.addWidget(self.velocitySallesBtn)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.velocityManBtn = QtWidgets.QRadioButton(self.centralwidget)
        self.velocityManBtn.setObjectName("velocityManBtn")
        self.velocityBtnGrp.addButton(self.velocityManBtn)
        self.horizontalLayout_5.addWidget(self.velocityManBtn)
        self.velocityManEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.velocityManEdit.setObjectName("velocityManEdit")
        self.horizontalLayout_5.addWidget(self.velocityManEdit)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_5.addWidget(self.label_3)
        self.verticalLayout_4.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_3.addLayout(self.verticalLayout_4)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout.addLayout(self.verticalLayout, 0, 1, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setRowCount(15)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setObjectName("tableWidget")
        self.verticalLayout_2.addWidget(self.tableWidget)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.importBtn = QtWidgets.QPushButton(self.centralwidget)
        self.importBtn.setObjectName("importBtn")
        self.horizontalLayout.addWidget(self.importBtn)
        self.saveBtn = QtWidgets.QPushButton(self.centralwidget)
        self.saveBtn.setObjectName("saveBtn")
        self.horizontalLayout.addWidget(self.saveBtn)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 795, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.loadData(select=0)


        self.importBtn.clicked.connect(self.loadData)
        self.saveBtn.clicked.connect(self.saveData)
        self.checkBox_createAnimation.isChecked

        if self.checkBox_createAnimation.isChecked() is False:
            self.numIterationSample_lineEdit.setWindowFlags(QtCore.Qt.ItemIsEditable)
            # self.tableWidget.item(row, 0).setFlags(QtCore.Qt.ItemIsEditable)

        self.runBtn.clicked.connect(self.runSim)

        self.progressBar_increment = 0
        self.progressBar_step = 0

    def runSim(self):
        self.progressBar.setValue(0)

        parameters = {}
        for i in range(self.tableWidget.rowCount()):
            parameters[self.tableWidget.item(i,0).text()] = eval(self.tableWidget.item(i,1).text())


        parameters['iterations'] = int(self.numiterationsEdit.text())

        if self.rupertBtn.isChecked():
            parameters['terrain'] = 'rupert'
        elif self.ShallowBtn.isChecked():
            parameters['terrain'] = 'river_shallow'
        else:
            parameters['terrain'] = 'river'

        parameters['velocity'] = 0
        if self.velocityManBtn.isChecked():
            parameters['velocity'] = eval(self.velocityManEdit.text())


        self.progressBar_increment = round(1.0/parameters['iterations']*100)

        # print(parameters)

        CAenv = main.CAenvironment(parameters)
        q_th0 = parameters['q_th[y,x]']
        q_cj0 = parameters['q_cj[y,x,0]']
        q_v0 = parameters['q_v[y,x]']


        from timeit import default_timer as timer
        start = timer()
        for i in range(parameters['iterations']):

            CAenv.addSource(q_th0,q_v0, q_cj0)
            # CAenv.add_source_constant(q_th0,q_v0, q_cj0)
            CAenv.CAtimeStep()
            ind = np.unravel_index(np.argmax(CAenv.grid.Q_th, axis=None), CAenv.grid.Q_th.shape)
            CAenv.head_velocity.append(CAenv.grid.Q_v[ind])
            self.incrementProgressBar()

            if ( (i+1) % int(self.numIterationSample_lineEdit.text()) == 0) and i > 0:
                CAenv.sampleValues()
                CAenv.printSubstates(i)
        CAenv.plotStabilityCurves(i)
        CAenv.writeToTxt(i)
        self.progressBar.setValue(100)
        print("time = ", timer() - start)



    def incrementProgressBar(self):
        self.progressBar_step += self.progressBar_increment
        self.progressBar.setValue(self.progressBar_step)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Test"))
        self.runBtn.setText(_translate("MainWindow", "Run simulation"))
        self.stopBtn.setText(_translate("MainWindow", "Stop simulation"))
        self.label_4.setText(_translate("MainWindow", "Include transition function component:"))
        self.checkBox_T1.setText(_translate("MainWindow", "T_1"))
        self.checkBox_T2.setText(_translate("MainWindow", "T_2"))
        self.checkBox_I1.setText(_translate("MainWindow", "I_1"))
        self.checkBox_I2.setText(_translate("MainWindow", "I_2"))
        self.checkBox_I3.setText(_translate("MainWindow", "I_3"))
        self.checkBox_I4.setText(_translate("MainWindow", "I_4"))
        self.checkBox_createAnimation.setText(_translate("MainWindow", "Create animation"))
        self.label_5.setText(_translate("MainWindow", "Iterations between each sample:"))
        self.numIterationSample_lineEdit.setText(_translate("MainWindow", "50"))
        self.Bathylabel.setText(_translate("MainWindow", "Select a bathymetry:"))
        self.rupertBtn.setText(_translate("MainWindow", "Rupert Inlet"))
        self.ShallowBtn.setText(_translate("MainWindow", "Shallow river"))
        self.SteeprBtn.setText(_translate("MainWindow", "Steep river"))
        self.label.setText(_translate("MainWindow", "Number of iterations:"))
        self.numiterationsEdit.setText(_translate("MainWindow", "500"))
        self.label_2.setText(_translate("MainWindow", "Sphere settling velocity:"))
        self.velocitySallesBtn.setText(_translate("MainWindow", "Salles\' equation"))
        self.velocityManBtn.setText(_translate("MainWindow", "Manual velocity:"))
        self.velocityManEdit.setText(_translate("MainWindow", "0.001"))
        self.label_3.setText(_translate("MainWindow", "m/s"))
        self.importBtn.setText(_translate("MainWindow", "Import settings"))
        self.saveBtn.setText(_translate("MainWindow", "Save settings"))



# class ThreadClass(QtCore.QThread):
#     def __init__(self, parent=None):
#         super(ThreadClass, self).__init__(parent)
#         self.signal = QtCore.pyqtSignal()
#
#
#     def run(self, parameters: dict, numIterationSample_lineEdit: int):
#         CAenv = main.CAenvironment(parameters)
#
#         for i in range(parameters['iterations']):
#             CAenv.addSource()
#             CAenv.CAtimeStep()
#             self.signal.emit()
#
#             # self.incrementProgressBar()
#
#
#             if ( (i+1) % int(numIterationSample_lineEdit) == 0) and i > 0:
#                 CAenv.printSubstates(i)
#                 CAenv.sampleValues()
#         CAenv.plotStabilityCurves(i)
#         CAenv.writeToTxt(i)



    # def setImage(self):
    #     fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp")
    #     if fileName:
    #         pixmap = QtGui.QPixmap(fileName)
    #         pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio)
    #         self.imageLbl.setPixmap(pixmap)
    #         self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    sys.exit(app.exec_())
