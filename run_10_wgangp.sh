mkdir results/cifar10_1/WGAN_GP/sliced
mkdir results/cifar10_2/WGAN_GP/sliced
mkdir results/cifar10_3/WGAN_GP/sliced
mkdir results/cifar10_4/WGAN_GP/sliced
mkdir results/cifar10_5/WGAN_GP/sliced
mkdir results/cifar10_6/WGAN_GP/sliced
mkdir results/cifar10_7/WGAN_GP/sliced
mkdir results/cifar10_8/WGAN_GP/sliced
mkdir results/cifar10_9/WGAN_GP/sliced
mkdir results/cifar10_10/WGAN_GP/sliced
python main.py --dataset cifar10_1 --gan_type WGAN_GP --epoch 100 --batch_size 64 --repeat 0 --input_size 64
python main.py --dataset cifar10_2 --gan_type WGAN_GP --epoch 100 --batch_size 64 --repeat 0 --input_size 64
python main.py --dataset cifar10_3 --gan_type WGAN_GP --epoch 100 --batch_size 64 --repeat 0 --input_size 64
python main.py --dataset cifar10_4 --gan_type WGAN_GP --epoch 100 --batch_size 64 --repeat 0 --input_size 64
python main.py --dataset cifar10_5 --gan_type WGAN_GP --epoch 100 --batch_size 64 --repeat 0 --input_size 64
python main.py --dataset cifar10_6 --gan_type WGAN_GP --epoch 300 --batch_size 64 --repeat 0 --input_size 64
python main.py --dataset cifar10_7 --gan_type WGAN_GP --epoch 300 --batch_size 64 --repeat 0 --input_size 64
python main.py --dataset cifar10_8 --gan_type WGAN_GP --epoch 300 --batch_size 64 --repeat 0 --input_size 64
python main.py --dataset cifar10_9 --gan_type WGAN_GP --epoch 300 --batch_size 64 --repeat 0 --input_size 64
python main.py --dataset cifar10_10 --gan_type WGAN_GP --epoch 300 --batch_size 64 --repeat 0 --input_size 64

cp results/slice_add.py results/cifar10_1/WGAN_GP/sliced 
cp results/slice_add.py results/cifar10_2/WGAN_GP/sliced
cp results/slice_add.py results/cifar10_3/WGAN_GP/sliced
cp results/slice_add.py results/cifar10_4/WGAN_GP/sliced
cp results/slice_add.py results/cifar10_5/WGAN_GP/sliced
cp results/slice_add.py results/cifar10_6/WGAN_GP/sliced
cp results/slice_add.py results/cifar10_7/WGAN_GP/sliced
cp results/slice_add.py results/cifar10_8/WGAN_GP/sliced
cp results/slice_add.py results/cifar10_9/WGAN_GP/sliced
cp results/slice_add.py results/cifar10_10/WGAN_GP/sliced


cd results/cifar10_1/WGAN_GP/sliced/
python slice_add.py 1
cd ../../../../results/cifar10_2/WGAN_GP/sliced/
python slice_add.py 2
cd ../../../../results/cifar10_3/WGAN_GP/sliced/
python slice_add.py 3
cd ../../../../results/cifar10_4/WGAN_GP/sliced/
python slice_add.py 4
cd ../../../../results/cifar10_5/WGAN_GP/sliced/
python slice_add.py 5
cd ../../../../results/cifar10_6/WGAN_GP/sliced/
python slice_add.py 6
cd ../../../../results/cifar10_7/WGAN_GP/sliced/
python slice_add.py 7
cd ../../../../results/cifar10_8/WGAN_GP/sliced/
python slice_add.py 8
cd ../../../../results/cifar10_9/WGAN_GP/sliced/
python slice_add.py 9
cd ../../../../results/cifar10_10/WGAN_GP/sliced/
python slice_add.py 10
cd ../../../../

cp -r results/cifar10_1 results/10_folder_results/5000_images/iteration1
cp -r results/cifar10_2 results/10_folder_results/5000_images/iteration1
cp -r results/cifar10_3 results/10_folder_results/5000_images/iteration1
cp -r results/cifar10_4 results/10_folder_results/5000_images/iteration1
cp -r results/cifar10_5 results/10_folder_results/5000_images/iteration1
cp -r results/cifar10_6 results/10_folder_results/5000_images/iteration1
cp -r results/cifar10_7 results/10_folder_results/5000_images/iteration1
cp -r results/cifar10_8 results/10_folder_results/5000_images/iteration1
cp -r results/cifar10_9 results/10_folder_results/5000_images/iteration1
cp -r results/cifar10_10 results/10_folder_results/5000_images/iteration1
cp -r model_result_distributed_1 results/10_folder_results/5000_images/iteration1
cp -r model_result_distributed_2 results/10_folder_results/5000_images/iteration1
cp -r model_result_distributed_3 results/10_folder_results/5000_images/iteration1
cp -r model_result_distributed_4 results/10_folder_results/5000_images/iteration1
cp -r model_result_distributed_5 results/10_folder_results/5000_images/iteration1
cp -r model_result_distributed_6 results/10_folder_results/5000_images/iteration1
cp -r model_result_distributed_7 results/10_folder_results/5000_images/iteration1
cp -r model_result_distributed_8 results/10_folder_results/5000_images/iteration1
cp -r model_result_distributed_9 results/10_folder_results/5000_images/iteration1
cp -r model_result_distributed_10 results/10_folder_results/5000_images/iteration1

python main.py --dataset cifar10_1 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_1/discriminator_epoch_099.pth --netG_path model_result_distributed_1/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_2 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_2/discriminator_epoch_099.pth --netG_path model_result_distributed_2/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_3 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_3/discriminator_epoch_099.pth --netG_path model_result_distributed_3/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_4 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_4/discriminator_epoch_099.pth --netG_path model_result_distributed_4/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_5 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_5/discriminator_epoch_099.pth --netG_path model_result_distributed_5/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_6 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_6/discriminator_epoch_099.pth --netG_path model_result_distributed_6/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_7 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_7/discriminator_epoch_099.pth --netG_path model_result_distributed_7/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_8 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_8/discriminator_epoch_099.pth --netG_path model_result_distributed_8/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_9 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_9/discriminator_epoch_099.pth --netG_path model_result_distributed_9/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_10 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_10/discriminator_epoch_099.pth --netG_path model_result_distributed_10/generator_epoch_099.pth --input_size 64

cd results/cifar10_1/WGAN_GP/sliced/
python slice_add.py 1
cd ../../../../results/cifar10_2/WGAN_GP/sliced/
python slice_add.py 2
cd ../../../../results/cifar10_3/WGAN_GP/sliced/
python slice_add.py 3
cd ../../../../results/cifar10_4/WGAN_GP/sliced/
python slice_add.py 4
cd ../../../../results/cifar10_5/WGAN_GP/sliced/
python slice_add.py 5
cd ../../../../results/cifar10_6/WGAN_GP/sliced/
python slice_add.py 6
cd ../../../../results/cifar10_7/WGAN_GP/sliced/
python slice_add.py 7
cd ../../../../results/cifar10_8/WGAN_GP/sliced/
python slice_add.py 8
cd ../../../../results/cifar10_9/WGAN_GP/sliced/
python slice_add.py 9
cd ../../../../results/cifar10_10/WGAN_GP/sliced/
python slice_add.py 10
cd ../../../../

cp -r results/cifar10_1 results/10_folder_results/5000_images/iteration2
cp -r results/cifar10_2 results/10_folder_results/5000_images/iteration2
cp -r results/cifar10_3 results/10_folder_results/5000_images/iteration2
cp -r results/cifar10_4 results/10_folder_results/5000_images/iteration2
cp -r results/cifar10_5 results/10_folder_results/5000_images/iteration2
cp -r results/cifar10_6 results/10_folder_results/5000_images/iteration2
cp -r results/cifar10_7 results/10_folder_results/5000_images/iteration2
cp -r results/cifar10_8 results/10_folder_results/5000_images/iteration2
cp -r results/cifar10_9 results/10_folder_results/5000_images/iteration2
cp -r results/cifar10_10 results/10_folder_results/5000_images/iteration2
cp -r model_result_distributed_1 results/10_folder_results/5000_images/iteration2
cp -r model_result_distributed_2 results/10_folder_results/5000_images/iteration2
cp -r model_result_distributed_3 results/10_folder_results/5000_images/iteration2
cp -r model_result_distributed_4 results/10_folder_results/5000_images/iteration2
cp -r model_result_distributed_5 results/10_folder_results/5000_images/iteration2
cp -r model_result_distributed_6 results/10_folder_results/5000_images/iteration2
cp -r model_result_distributed_7 results/10_folder_results/5000_images/iteration2
cp -r model_result_distributed_8 results/10_folder_results/5000_images/iteration2
cp -r model_result_distributed_9 results/10_folder_results/5000_images/iteration2
cp -r model_result_distributed_10 results/10_folder_results/5000_images/iteration2

python main.py --dataset cifar10_1 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_1/discriminator_epoch_099.pth --netG_path model_result_distributed_1/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_2 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_2/discriminator_epoch_099.pth --netG_path model_result_distributed_2/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_3 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_3/discriminator_epoch_099.pth --netG_path model_result_distributed_3/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_4 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_4/discriminator_epoch_099.pth --netG_path model_result_distributed_4/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_5 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_5/discriminator_epoch_099.pth --netG_path model_result_distributed_5/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_6 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_6/discriminator_epoch_099.pth --netG_path model_result_distributed_6/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_7 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_7/discriminator_epoch_099.pth --netG_path model_result_distributed_7/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_8 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_8/discriminator_epoch_099.pth --netG_path model_result_distributed_8/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_9 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_9/discriminator_epoch_099.pth --netG_path model_result_distributed_9/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_10 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_10/discriminator_epoch_099.pth --netG_path model_result_distributed_10/generator_epoch_099.pth --input_size 64

cd results/cifar10_1/WGAN_GP/sliced/
python slice_add.py 1
cd ../../../../results/cifar10_2/WGAN_GP/sliced/
python slice_add.py 2
cd ../../../../results/cifar10_3/WGAN_GP/sliced/
python slice_add.py 3
cd ../../../../results/cifar10_4/WGAN_GP/sliced/
python slice_add.py 4
cd ../../../../results/cifar10_5/WGAN_GP/sliced/
python slice_add.py 5
cd ../../../../results/cifar10_6/WGAN_GP/sliced/
python slice_add.py 6
cd ../../../../results/cifar10_7/WGAN_GP/sliced/
python slice_add.py 7
cd ../../../../results/cifar10_8/WGAN_GP/sliced/
python slice_add.py 8
cd ../../../../results/cifar10_9/WGAN_GP/sliced/
python slice_add.py 9
cd ../../../../results/cifar10_10/WGAN_GP/sliced/
python slice_add.py 10
cd ../../../../

cp -r results/cifar10_1 results/10_folder_results/5000_images/iteration3
cp -r results/cifar10_2 results/10_folder_results/5000_images/iteration3
cp -r results/cifar10_3 results/10_folder_results/5000_images/iteration3
cp -r results/cifar10_4 results/10_folder_results/5000_images/iteration3
cp -r results/cifar10_5 results/10_folder_results/5000_images/iteration3
cp -r results/cifar10_6 results/10_folder_results/5000_images/iteration3
cp -r results/cifar10_7 results/10_folder_results/5000_images/iteration3
cp -r results/cifar10_8 results/10_folder_results/5000_images/iteration3
cp -r results/cifar10_9 results/10_folder_results/5000_images/iteration3
cp -r results/cifar10_10 results/10_folder_results/5000_images/iteration3
cp -r model_result_distributed_1 results/10_folder_results/5000_images/iteration3
cp -r model_result_distributed_2 results/10_folder_results/5000_images/iteration3
cp -r model_result_distributed_3 results/10_folder_results/5000_images/iteration3
cp -r model_result_distributed_4 results/10_folder_results/5000_images/iteration3
cp -r model_result_distributed_5 results/10_folder_results/5000_images/iteration3
cp -r model_result_distributed_6 results/10_folder_results/5000_images/iteration3
cp -r model_result_distributed_7 results/10_folder_results/5000_images/iteration3
cp -r model_result_distributed_8 results/10_folder_results/5000_images/iteration3
cp -r model_result_distributed_9 results/10_folder_results/5000_images/iteration3
cp -r model_result_distributed_10 results/10_folder_results/5000_images/iteration3

python main.py --dataset cifar10_1 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_1/discriminator_epoch_099.pth --netG_path model_result_distributed_1/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_2 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_2/discriminator_epoch_099.pth --netG_path model_result_distributed_2/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_3 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_3/discriminator_epoch_099.pth --netG_path model_result_distributed_3/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_4 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_4/discriminator_epoch_099.pth --netG_path model_result_distributed_4/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_5 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_5/discriminator_epoch_099.pth --netG_path model_result_distributed_5/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_6 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_6/discriminator_epoch_099.pth --netG_path model_result_distributed_6/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_7 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_7/discriminator_epoch_099.pth --netG_path model_result_distributed_7/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_8 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_8/discriminator_epoch_099.pth --netG_path model_result_distributed_8/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_9 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_9/discriminator_epoch_099.pth --netG_path model_result_distributed_9/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_10 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_10/discriminator_epoch_099.pth --netG_path model_result_distributed_10/generator_epoch_099.pth --input_size 64


cd results/cifar10_1/WGAN_GP/sliced/
python slice_add.py 1
cd ../../../../results/cifar10_2/WGAN_GP/sliced/
python slice_add.py 2
cd ../../../../results/cifar10_3/WGAN_GP/sliced/
python slice_add.py 3
cd ../../../../results/cifar10_4/WGAN_GP/sliced/
python slice_add.py 4
cd ../../../../results/cifar10_5/WGAN_GP/sliced/
python slice_add.py 5
cd ../../../../results/cifar10_6/WGAN_GP/sliced/
python slice_add.py 6
cd ../../../../results/cifar10_7/WGAN_GP/sliced/
python slice_add.py 7
cd ../../../../results/cifar10_8/WGAN_GP/sliced/
python slice_add.py 8
cd ../../../../results/cifar10_9/WGAN_GP/sliced/
python slice_add.py 9
cd ../../../../results/cifar10_10/WGAN_GP/sliced/
python slice_add.py 10
cd ../../../../

cp -r results/cifar10_1 results/10_folder_results/5000_images/iteration4
cp -r results/cifar10_2 results/10_folder_results/5000_images/iteration4
cp -r results/cifar10_3 results/10_folder_results/5000_images/iteration4
cp -r results/cifar10_4 results/10_folder_results/5000_images/iteration4
cp -r results/cifar10_5 results/10_folder_results/5000_images/iteration4
cp -r results/cifar10_6 results/10_folder_results/5000_images/iteration4
cp -r results/cifar10_7 results/10_folder_results/5000_images/iteration4
cp -r results/cifar10_8 results/10_folder_results/5000_images/iteration4
cp -r results/cifar10_9 results/10_folder_results/5000_images/iteration4
cp -r results/cifar10_10 results/10_folder_results/5000_images/iteration4
cp -r model_result_distributed_1 results/10_folder_results/5000_images/iteration4
cp -r model_result_distributed_2 results/10_folder_results/5000_images/iteration4
cp -r model_result_distributed_3 results/10_folder_results/5000_images/iteration4
cp -r model_result_distributed_4 results/10_folder_results/5000_images/iteration4
cp -r model_result_distributed_5 results/10_folder_results/5000_images/iteration4
cp -r model_result_distributed_6 results/10_folder_results/5000_images/iteration4
cp -r model_result_distributed_7 results/10_folder_results/5000_images/iteration4
cp -r model_result_distributed_8 results/10_folder_results/5000_images/iteration4
cp -r model_result_distributed_9 results/10_folder_results/5000_images/iteration4
cp -r model_result_distributed_10 results/10_folder_results/5000_images/iteration4

python main.py --dataset cifar10_1 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_1/discriminator_epoch_099.pth --netG_path model_result_distributed_1/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_2 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_2/discriminator_epoch_099.pth --netG_path model_result_distributed_2/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_3 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_3/discriminator_epoch_099.pth --netG_path model_result_distributed_3/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_4 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_4/discriminator_epoch_099.pth --netG_path model_result_distributed_4/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_5 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_5/discriminator_epoch_099.pth --netG_path model_result_distributed_5/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_6 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_6/discriminator_epoch_099.pth --netG_path model_result_distributed_6/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_7 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_7/discriminator_epoch_099.pth --netG_path model_result_distributed_7/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_8 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_8/discriminator_epoch_099.pth --netG_path model_result_distributed_8/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_9 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_9/discriminator_epoch_099.pth --netG_path model_result_distributed_9/generator_epoch_099.pth --input_size 64
python main.py --dataset cifar10_10 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_10/discriminator_epoch_099.pth --netG_path model_result_distributed_10/generator_epoch_099.pth --input_size 64


cd results/cifar10_1/WGAN_GP/sliced/
python slice_add.py 1
cd ../../../../results/cifar10_2/WGAN_GP/sliced/
python slice_add.py 2
cd ../../../../results/cifar10_3/WGAN_GP/sliced/
python slice_add.py 3
cd ../../../../results/cifar10_4/WGAN_GP/sliced/
python slice_add.py 4
cd ../../../../results/cifar10_5/WGAN_GP/sliced/
python slice_add.py 5
cd ../../../../results/cifar10_6/WGAN_GP/sliced/
python slice_add.py 6
cd ../../../../results/cifar10_7/WGAN_GP/sliced/
python slice_add.py 7
cd ../../../../results/cifar10_8/WGAN_GP/sliced/
python slice_add.py 8
cd ../../../../results/cifar10_9/WGAN_GP/sliced/
python slice_add.py 9
cd ../../../../results/cifar10_10/WGAN_GP/sliced/
python slice_add.py 10
cd ../../../../

cp -r results/cifar10_1 results/10_folder_results/5000_images/iteration5
cp -r results/cifar10_2 results/10_folder_results/5000_images/iteration5
cp -r results/cifar10_3 results/10_folder_results/5000_images/iteration5
cp -r results/cifar10_4 results/10_folder_results/5000_images/iteration5
cp -r results/cifar10_5 results/10_folder_results/5000_images/iteration5
cp -r results/cifar10_6 results/10_folder_results/5000_images/iteration5
cp -r results/cifar10_7 results/10_folder_results/5000_images/iteration5
cp -r results/cifar10_8 results/10_folder_results/5000_images/iteration5
cp -r results/cifar10_9 results/10_folder_results/5000_images/iteration5
cp -r results/cifar10_10 results/10_folder_results/5000_images/iteration5
cp -r model_result_distributed_1 results/10_folder_results/5000_images/iteration5
cp -r model_result_distributed_2 results/10_folder_results/5000_images/iteration5
cp -r model_result_distributed_3 results/10_folder_results/5000_images/iteration5
cp -r model_result_distributed_4 results/10_folder_results/5000_images/iteration5
cp -r model_result_distributed_5 results/10_folder_results/5000_images/iteration5
cp -r model_result_distributed_6 results/10_folder_results/5000_images/iteration5
cp -r model_result_distributed_7 results/10_folder_results/5000_images/iteration5
cp -r model_result_distributed_8 results/10_folder_results/5000_images/iteration5
cp -r model_result_distributed_9 results/10_folder_results/5000_images/iteration5
cp -r model_result_distributed_10 results/10_folder_results/5000_images/iteration5



