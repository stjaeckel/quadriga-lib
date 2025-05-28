function test_arrayant_export_obj_file

ant = quadriga_lib.arrayant_generate('3GPP');
quadriga_lib.arrayant_export_obj_file('test_mex.obj',ant);

assertTrue(exist( 'test_mex.obj','file' )==2);
assertTrue(exist( 'test_mex.mtl','file' )==2);

delete('test_mex.obj');
delete('test_mex.mtl');

end