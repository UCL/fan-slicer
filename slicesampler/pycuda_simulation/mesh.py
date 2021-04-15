# coding=utf-8

""" Module with mesh class """

import numpy as np
import vtk
import vtk.numpy_interface.dataset_adapter as dsa
from vtk.util.numpy_support import vtk_to_numpy


class Mesh3D:
    """
    Class to process mesh
    """
    def __init__(self, v, t):
        """
        Constructs mesh with vertices v
        and faces t
        """
        self.vertices = v
        self.faces = t

    def get_bounding_box(self):
        """
        Gets bounding box of Mesh3D object
        :return: bounding box of mesh3D object
        """
        bound_box = None
        if self.vertices is not None and self.faces is not None:
            upper_bound = np.array([np.amax(self.vertices[:, 0]),
                                    np.amax(self.vertices[:, 1]),
                                    np.amax(self.vertices[:, 2])])

            lower_bound = np.array([np.amin(self.vertices[:, 0]),
                                    np.amin(self.vertices[:, 1]),
                                    np.amin(self.vertices[:, 2])])

            bound_box = np.vstack([lower_bound, upper_bound])

        return bound_box


def join_mesh(mesh1, mesh2):
    """
    Merges two input Mesh3D objects
    mesh1 and mesh2

    :param mesh1: first mesh3D
    :param mesh2: second mesh3D
    :return: joint mesh3D object
    """
    # Join vertices
    joint_vertices = np.vstack((mesh1.vertices,
                                mesh2.vertices))
    # Join triangles by getting the number of faces
    # of mesh1
    num_faces1 = mesh1.vertices.shape[0]
    joint_faces = np.vstack((mesh1.faces,
                            mesh2.faces+num_faces1))
    # Create joint mesh
    return Mesh3D(joint_vertices, joint_faces)


def load_mesh_from_vtk(vtk_dir):
    """
    Creates Mesh3D object from an input
    vtk polydata file

    :param vtk_dir: vtk file
    :return: Mesh3D object
    """
    # Set VTK reader
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_dir)
    reader.Update()
    polydata = reader.GetOutput()
    # Extract faces and vertices
    vertices = vtk_to_numpy(dsa.WrapDataObject(polydata).Points)
    faces = vtk_to_numpy(dsa.WrapDataObject(polydata).Polygons)
    num_faces = np.int(faces.shape[0]/4)

    # Reshape faces (VTK array comes with
    # (4 elements per polygon -> Num of vertices, x, y, z)
    faces = faces.reshape(num_faces, 4)
    faces = faces[:, 1:]

    # Create and output Mesh3D object
    return Mesh3D(vertices, faces)
