# -*- coding: utf-8 -*-
"""
/***************************************************************************
 photopreprocessing
                                 A QGIS plugin
 Photo Preprocessing
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2024-04-30
        git sha              : $Format:%H$
        copyright            : (C) 2024 by cast
        email                : cast@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from pyodm import Node
import threading
# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .photo_preprocessing_dialog import photopreprocessingDialog
import os.path
from PyQt5.QtWidgets import QWidget,  QDialog,  QMainWindow, QLabel
import os
from PyQt5.QtCore import Qt, pyqtSignal
from qgis.core import QgsProject,  QgsRasterLayer
import psycopg2
import json
from datetime import datetime
import serverdatabaseconnection.ServerDataBaseConnection as DB_Server
from serverdatabaseconnection.ServerDataBaseConnection import create_db_path, set_db_connection
import helpers
from helpers import read_json, show_warning_message

database_path = helpers.get_dataset_path()
server_path = helpers.get_server_path()
config_path = helpers.get_config_path()
db_queries_path = helpers.get_db_queries_path()


class LoadingWindow(QMainWindow):
    window_closed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loading...")
        self.setFixedSize(200, 100)
        self.label = QLabel("Processing task...", self)
        self.label.setGeometry(10, 10, 180, 80)
        self.label.setAlignment(Qt.AlignCenter)

    def closeEvent(self, event):
        # Emit the signal when the window is closed
        self.window_closed.emit()
        event.accept()


class photopreprocessing:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'photopreprocessing_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&Photo Preprocessing')
        self.loading_window = LoadingWindow()
        self.ready_to_save = False
        self.loading_window.window_closed.connect(self.loading_window_closed)
        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.first_start = None

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('photopreprocessing', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/photo_preprocessing/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u''),
            callback=self.run,
            parent=self.iface.mainWindow())

        # will be set False in run()
        self.first_start = True

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Photo Preprocessing'),
                action)
            self.iface.removeToolBarIcon(action)

    @staticmethod
    def is_image(filename):
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff']
        return any(filename.lower().endswith(ext) for ext in image_extensions)

    @staticmethod
    def contains_only_images(filenames):
        for filename in filenames:
            if not photopreprocessing.is_image(filename):
                return False
        return True

    @staticmethod
    def all_images_have_same_extension(filenames):
        extensions = set()
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            extensions.add(ext.lower())
        return len(extensions) == 1

    def create_or_set_db_connection(self):
        if not os.path.exists(database_path):
            create_db_path(database_path)
            self.open_add_db_info_dialog()

        res = set_db_connection(read_json(database_path))
        if isinstance(res, psycopg2.OperationalError):
            show_warning_message("Change or Set DB connection")
            return False
        else:
            self.cursor, self.conn = res
            return True

    def open_add_db_info_dialog(self):
        self.add_db_info_dialog = DB_Server.AddDatabaseInfo()
        if self.add_db_info_dialog.exec_() == QDialog.Accepted:
            print("done")

    def open_add_server_info_dialog(self):
        self.add_server_info_dialog = DB_Server.AddServerInfo()
        if self.add_server_info_dialog.exec_() == QDialog.Accepted:
            print("done")

    def save_to_db(self, map_name):
        if not os.path.exists(database_path):
            create_db_path(database_path)
            self.open_add_db_info_dialog()
        res = self.create_or_set_db_connection()
        if not res:
            show_warning_message("Orthophoto is not saved in DB")
            return
        try:
            with open(os.path.join(self.dlg.path_to_save_map.text(),\
                    "odm_orthophoto/odm_orthophoto.tif"), 'rb') as f:
                image_data = f.read()
        except Exception as e:
            show_warning_message(e)
            return
        db_info = read_json(database_path)
        try:
            cursor = self.cursor
            queries = read_json(db_queries_path)
            insert_query = queries["add_plugin_result"]
            cursor.execute(insert_query, (db_info["user"], "mapcreation", map_name, datetime.now(), image_data))
            self.conn.commit()
            cursor.close()
            self.conn.close()
            self.ready_to_save = True
            return True
        except Exception as e:
            show_warning_message(e)
            return e

    def loading_window_closed(self):
        if self.ready_to_save:
            photopreprocessing.open_in_qgis(os.path.join(self.dlg.path_to_save_map.text(), \
                                                         "odm_orthophoto/odm_orthophoto.tif"))
            if self.dlg.save_map_to_db.isChecked():
                output_text = self.dlg.path_to_save_map.text()
                if self.dlg.name_of_map.text() =="Default: Directory name":
                    if "/" in output_text:
                        last_slash_index = output_text.rfind("/")
                        map_name = output_text[last_slash_index + 1:]
                    else:
                        map_name = output_text
                else:
                    map_name = self.dlg.name_of_map.text()
                self.save_to_db(map_name)

    @staticmethod
    def get_files_from_dir(directory_path):
        try:
            contents = os.listdir(directory_path)
            contents = [os.path.join(directory_path, item) for item in contents]
        except FileNotFoundError as e:
            return e
        return contents

    def generate_map(self, content):

        try:
            threading.Thread(target=self.process_task, args=(content, self.loading_window)).start()
        except Exception as e:
            print("Error:", e)
            print("Failed to start processing task.")
            return e
        return True

    def process_task(self, content, loading_window):
        try:
            if not os.path.exists(server_path):
                DB_Server.create_server_path(server_path)
                self.open_add_server_info_dialog()
            server_info = read_json(server_path)
            self.loading_window.show()
            n = Node(server_info["SERVER_HOST"], server_info["SERVER_PORT"])
            task = n.create_task(content, {'fast-orthophoto': True})
            task.wait_for_completion()
            task.download_assets(self.dlg.path_to_save_map.text())
        except Exception as e:
            loading_window.close()  # Close loading window when task is done
            print("Error:", e)
            print("Failed to process task.")
            show_warning_message(e)
            return e
        loading_window.close()  # Close loading window when task is done
        return True

    @staticmethod
    def open_in_qgis(image_path):
        raster_layer = QgsRasterLayer(image_path, "output")
        if not raster_layer.isValid():
            print("Error: Unable to load raster layer.")
        else:
            QgsProject.instance().addMapLayer(raster_layer)

    def run(self):
        """Run method that performs all the real work"""

        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        self.dlg = photopreprocessingDialog()

        # show the dialog
        self.dlg.show()
        self.dlg.save_map_to_db.stateChanged.connect(self.toggleLineEdit)
        result = self.dlg.exec_()
        if result:
            input_path = self.dlg.path_to_image_dir.text()
            res = photopreprocessing.get_files_from_dir(input_path)
            if isinstance(res, FileNotFoundError):
                show_warning_message(res)
                return
            else:
                content = res
                res = photopreprocessing.contains_only_images(content)
                if not res:
                    show_warning_message("All files should be images in\
                     the directory!")
                    return
                res = photopreprocessing.all_images_have_same_extension(content)
                if not res:
                    show_warning_message("All images should have the same \
                                                            extension!")
                    return
                res = self.generate_map(content)
                if not res:
                    show_warning_message(res)
                    return

    def toggleLineEdit(self, checked):
        # Enable name_of_map line edit when save_map_to_db checkbox is checked
        self.dlg.name_of_map.setEnabled(checked)

