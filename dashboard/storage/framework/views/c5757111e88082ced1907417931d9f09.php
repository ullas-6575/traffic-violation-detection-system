<?php if (isset($component)) { $__componentOriginal9ac128a9029c0e4701924bd2d73d7f54 = $component; } ?>
<?php if (isset($attributes)) { $__attributesOriginal9ac128a9029c0e4701924bd2d73d7f54 = $attributes; } ?>
<?php $component = App\View\Components\AppLayout::resolve([] + (isset($attributes) && $attributes instanceof Illuminate\View\ComponentAttributeBag ? $attributes->all() : [])); ?>
<?php $component->withName('app-layout'); ?>
<?php if ($component->shouldRender()): ?>
<?php $__env->startComponent($component->resolveView(), $component->data()); ?>
<?php if (isset($attributes) && $attributes instanceof Illuminate\View\ComponentAttributeBag): ?>
<?php $attributes = $attributes->except(\App\View\Components\AppLayout::ignoredParameterNames()); ?>
<?php endif; ?>
<?php $component->withAttributes([]); ?>
     <?php $__env->slot('header', null, []); ?> 
        <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
                <a href="<?php echo e(route('violations.index')); ?>" class="text-sm font-semibold text-red-700 hover:text-red-900">← Back to violations</a>
                <h2 class="mt-1 text-2xl font-bold text-slate-900">Violation evidence</h2>
                <p class="font-mono text-xs text-slate-500"><?php echo e($violation->event_id); ?></p>
            </div>
            <span class="inline-flex self-start rounded-full px-3 py-1.5 text-xs font-bold uppercase tracking-wide <?php echo e(match($violation->status) {
                'CONFIRMED' => 'bg-emerald-100 text-emerald-800',
                'DISMISSED' => 'bg-slate-200 text-slate-700',
                default => 'bg-indigo-100 text-indigo-800',
            }); ?>"><?php echo e($violation->status); ?></span>
        </div>
     <?php $__env->endSlot(); ?>

    <div class="py-8">
        <div class="mx-auto max-w-7xl space-y-6 px-4 sm:px-6 lg:px-8">
            <?php if(session('status')): ?>
                <div class="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-medium text-emerald-800">
                    <?php echo e(session('status')); ?>

                </div>
            <?php endif; ?>

            <div class="grid gap-6 xl:grid-cols-3">
                <div class="space-y-6 xl:col-span-2">
                    <section class="overflow-hidden rounded-xl border border-slate-200 bg-slate-950 shadow-sm">
                        <div class="flex items-center justify-between border-b border-white/10 px-5 py-3 text-white">
                            <h3 class="font-semibold">Violation frame</h3>
                            <span class="text-xs text-slate-300">Frame <?php echo e(number_format($violation->frame_number)); ?></span>
                        </div>
                        <?php if($violation->image_path): ?>
                            <a href="<?php echo e(asset('storage/'.$violation->image_path)); ?>" target="_blank" rel="noopener">
                                <img src="<?php echo e(asset('storage/'.$violation->image_path)); ?>" alt="Full violation evidence for <?php echo e($violation->event_id); ?>"
                                     class="max-h-[640px] w-full object-contain">
                            </a>
                        <?php else: ?>
                            <div class="flex min-h-80 items-center justify-center text-sm text-slate-400">No full-frame evidence available</div>
                        <?php endif; ?>
                    </section>

                    <section class="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                        <h3 class="text-lg font-bold text-slate-900">Incident metadata</h3>
                        <dl class="mt-5 grid gap-x-8 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
                            <?php
                                $metadata = [
                                    ['Violation', str_replace('_', ' ', $violation->violation_type)],
                                    ['Captured', $violation->frame_timestamp?->format('d M Y, H:i:s')],
                                    ['Track ID', '#'.$violation->track_id],
                                    ['Speed', $violation->speed !== null ? number_format($violation->speed, 1).' km/h' : 'Not measured'],
                                    ['Speed limit', $violation->speed_limit !== null ? number_format($violation->speed_limit, 0).' km/h' : 'N/A'],
                                    ['Signal', $violation->signal_state ?: 'N/A'],
                                    ['Direction', $violation->direction ?: 'Unknown'],
                                    ['Vehicle color', ucfirst(strtolower($violation->vehicle_color)).' · '.number_format(($violation->color_confidence ?? 0) * 100, 1).'%'],
                                    ['OCR engine', $violation->ocr_engine ?: 'none'],
                                    ['OCR confidence', number_format(($violation->ocr_confidence ?? 0) * 100, 1).'%'],
                                    ['Raw OCR', $violation->ocr_raw_text ?: 'No text detected'],
                                    ['Database ID', '#'.$violation->id],
                                ];
                            ?>
                            <?php $__currentLoopData = $metadata; $__env->addLoop($__currentLoopData); foreach($__currentLoopData as [$label, $value]): $__env->incrementLoopIndices(); $loop = $__env->getLastLoop(); ?>
                                <div>
                                    <dt class="text-xs font-semibold uppercase tracking-wide text-slate-400"><?php echo e($label); ?></dt>
                                    <dd class="mt-1 break-words text-sm font-medium text-slate-800"><?php echo e($value); ?></dd>
                                </div>
                            <?php endforeach; $__env->popLoop(); $loop = $__env->getLastLoop(); ?>
                        </dl>
                    </section>
                </div>

                <aside class="space-y-6">
                    <section class="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                        <div class="flex items-start justify-between gap-4">
                            <div>
                                <p class="text-xs font-semibold uppercase tracking-wide text-slate-400">Recognized plate</p>
                                <p class="mt-1 text-xl font-bold text-slate-900"><?php echo e($violation->plate_number); ?></p>
                                <?php if($violation->original_plate_number): ?>
                                    <p class="mt-1 text-xs text-slate-500">Original: <?php echo e($violation->original_plate_number); ?></p>
                                <?php endif; ?>
                            </div>
                            <span class="rounded-md bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-600">
                                <?php echo e(number_format(($violation->ocr_confidence ?? 0) * 100, 1)); ?>%
                            </span>
                        </div>
                        <?php if($violation->plate_crop_path): ?>
                            <a href="<?php echo e(asset('storage/'.$violation->plate_crop_path)); ?>" target="_blank" rel="noopener">
                                <img src="<?php echo e(asset('storage/'.$violation->plate_crop_path)); ?>" alt="Number plate crop"
                                     class="mt-4 max-h-48 w-full rounded-lg bg-slate-100 object-contain ring-1 ring-slate-200">
                            </a>
                        <?php else: ?>
                            <div class="mt-4 flex h-32 items-center justify-center rounded-lg bg-slate-100 text-sm text-slate-400">No plate crop</div>
                        <?php endif; ?>
                    </section>

                    <section class="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                        <h3 class="text-lg font-bold text-slate-900">Officer review</h3>
                        <p class="mt-1 text-sm text-slate-500">Correct the plate if needed, then record a decision.</p>

                        <form method="POST" action="<?php echo e(route('violations.update', $violation)); ?>" class="mt-5 space-y-4">
                            <?php echo csrf_field(); ?>
                            <?php echo method_field('PATCH'); ?>

                            <div>
                                <label for="plate_number" class="block text-sm font-semibold text-slate-700">Plate number</label>
                                <input id="plate_number" name="plate_number" value="<?php echo e(old('plate_number', $violation->plate_number)); ?>" required maxlength="100"
                                       class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-red-500 focus:ring-red-500">
                                <?php if (isset($component)) { $__componentOriginalf94ed9c5393ef72725d159fe01139746 = $component; } ?>
<?php if (isset($attributes)) { $__attributesOriginalf94ed9c5393ef72725d159fe01139746 = $attributes; } ?>
<?php $component = Illuminate\View\AnonymousComponent::resolve(['view' => 'components.input-error','data' => ['messages' => $errors->get('plate_number'),'class' => 'mt-2']] + (isset($attributes) && $attributes instanceof Illuminate\View\ComponentAttributeBag ? $attributes->all() : [])); ?>
<?php $component->withName('input-error'); ?>
<?php if ($component->shouldRender()): ?>
<?php $__env->startComponent($component->resolveView(), $component->data()); ?>
<?php if (isset($attributes) && $attributes instanceof Illuminate\View\ComponentAttributeBag): ?>
<?php $attributes = $attributes->except(\Illuminate\View\AnonymousComponent::ignoredParameterNames()); ?>
<?php endif; ?>
<?php $component->withAttributes(['messages' => \Illuminate\View\Compilers\BladeCompiler::sanitizeComponentAttribute($errors->get('plate_number')),'class' => 'mt-2']); ?>
<?php echo $__env->renderComponent(); ?>
<?php endif; ?>
<?php if (isset($__attributesOriginalf94ed9c5393ef72725d159fe01139746)): ?>
<?php $attributes = $__attributesOriginalf94ed9c5393ef72725d159fe01139746; ?>
<?php unset($__attributesOriginalf94ed9c5393ef72725d159fe01139746); ?>
<?php endif; ?>
<?php if (isset($__componentOriginalf94ed9c5393ef72725d159fe01139746)): ?>
<?php $component = $__componentOriginalf94ed9c5393ef72725d159fe01139746; ?>
<?php unset($__componentOriginalf94ed9c5393ef72725d159fe01139746); ?>
<?php endif; ?>
                            </div>

                            <div>
                                <label for="status" class="block text-sm font-semibold text-slate-700">Decision</label>
                                <select id="status" name="status" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-red-500 focus:ring-red-500">
                                    <?php $__currentLoopData = ['PENDING', 'CONFIRMED', 'DISMISSED']; $__env->addLoop($__currentLoopData); foreach($__currentLoopData as $status): $__env->incrementLoopIndices(); $loop = $__env->getLastLoop(); ?>
                                        <option value="<?php echo e($status); ?>" <?php if(old('status', $violation->status) === $status): echo 'selected'; endif; ?>><?php echo e(ucfirst(strtolower($status))); ?></option>
                                    <?php endforeach; $__env->popLoop(); $loop = $__env->getLastLoop(); ?>
                                </select>
                                <?php if (isset($component)) { $__componentOriginalf94ed9c5393ef72725d159fe01139746 = $component; } ?>
<?php if (isset($attributes)) { $__attributesOriginalf94ed9c5393ef72725d159fe01139746 = $attributes; } ?>
<?php $component = Illuminate\View\AnonymousComponent::resolve(['view' => 'components.input-error','data' => ['messages' => $errors->get('status'),'class' => 'mt-2']] + (isset($attributes) && $attributes instanceof Illuminate\View\ComponentAttributeBag ? $attributes->all() : [])); ?>
<?php $component->withName('input-error'); ?>
<?php if ($component->shouldRender()): ?>
<?php $__env->startComponent($component->resolveView(), $component->data()); ?>
<?php if (isset($attributes) && $attributes instanceof Illuminate\View\ComponentAttributeBag): ?>
<?php $attributes = $attributes->except(\Illuminate\View\AnonymousComponent::ignoredParameterNames()); ?>
<?php endif; ?>
<?php $component->withAttributes(['messages' => \Illuminate\View\Compilers\BladeCompiler::sanitizeComponentAttribute($errors->get('status')),'class' => 'mt-2']); ?>
<?php echo $__env->renderComponent(); ?>
<?php endif; ?>
<?php if (isset($__attributesOriginalf94ed9c5393ef72725d159fe01139746)): ?>
<?php $attributes = $__attributesOriginalf94ed9c5393ef72725d159fe01139746; ?>
<?php unset($__attributesOriginalf94ed9c5393ef72725d159fe01139746); ?>
<?php endif; ?>
<?php if (isset($__componentOriginalf94ed9c5393ef72725d159fe01139746)): ?>
<?php $component = $__componentOriginalf94ed9c5393ef72725d159fe01139746; ?>
<?php unset($__componentOriginalf94ed9c5393ef72725d159fe01139746); ?>
<?php endif; ?>
                            </div>

                            <div>
                                <label for="officer_notes" class="block text-sm font-semibold text-slate-700">Officer notes</label>
                                <textarea id="officer_notes" name="officer_notes" rows="4" maxlength="2000" placeholder="Add observations or reasons for the decision…"
                                          class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-red-500 focus:ring-red-500"><?php echo e(old('officer_notes', $violation->officer_notes)); ?></textarea>
                                <?php if (isset($component)) { $__componentOriginalf94ed9c5393ef72725d159fe01139746 = $component; } ?>
<?php if (isset($attributes)) { $__attributesOriginalf94ed9c5393ef72725d159fe01139746 = $attributes; } ?>
<?php $component = Illuminate\View\AnonymousComponent::resolve(['view' => 'components.input-error','data' => ['messages' => $errors->get('officer_notes'),'class' => 'mt-2']] + (isset($attributes) && $attributes instanceof Illuminate\View\ComponentAttributeBag ? $attributes->all() : [])); ?>
<?php $component->withName('input-error'); ?>
<?php if ($component->shouldRender()): ?>
<?php $__env->startComponent($component->resolveView(), $component->data()); ?>
<?php if (isset($attributes) && $attributes instanceof Illuminate\View\ComponentAttributeBag): ?>
<?php $attributes = $attributes->except(\Illuminate\View\AnonymousComponent::ignoredParameterNames()); ?>
<?php endif; ?>
<?php $component->withAttributes(['messages' => \Illuminate\View\Compilers\BladeCompiler::sanitizeComponentAttribute($errors->get('officer_notes')),'class' => 'mt-2']); ?>
<?php echo $__env->renderComponent(); ?>
<?php endif; ?>
<?php if (isset($__attributesOriginalf94ed9c5393ef72725d159fe01139746)): ?>
<?php $attributes = $__attributesOriginalf94ed9c5393ef72725d159fe01139746; ?>
<?php unset($__attributesOriginalf94ed9c5393ef72725d159fe01139746); ?>
<?php endif; ?>
<?php if (isset($__componentOriginalf94ed9c5393ef72725d159fe01139746)): ?>
<?php $component = $__componentOriginalf94ed9c5393ef72725d159fe01139746; ?>
<?php unset($__componentOriginalf94ed9c5393ef72725d159fe01139746); ?>
<?php endif; ?>
                            </div>

                            <button class="w-full rounded-lg bg-red-700 px-4 py-3 text-sm font-bold text-white shadow-sm hover:bg-red-800 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2">
                                Save evidence review
                            </button>
                        </form>

                        <?php if($violation->reviewer): ?>
                            <div class="mt-5 border-t border-slate-200 pt-4 text-xs text-slate-500">
                                Last handled by <strong class="text-slate-700"><?php echo e($violation->reviewer->name); ?></strong>
                                <?php if($violation->reviewed_at): ?>
                                    on <?php echo e($violation->reviewed_at->format('d M Y, H:i')); ?>

                                <?php endif; ?>
                            </div>
                        <?php endif; ?>
                    </section>
                </aside>
            </div>
        </div>
    </div>
 <?php echo $__env->renderComponent(); ?>
<?php endif; ?>
<?php if (isset($__attributesOriginal9ac128a9029c0e4701924bd2d73d7f54)): ?>
<?php $attributes = $__attributesOriginal9ac128a9029c0e4701924bd2d73d7f54; ?>
<?php unset($__attributesOriginal9ac128a9029c0e4701924bd2d73d7f54); ?>
<?php endif; ?>
<?php if (isset($__componentOriginal9ac128a9029c0e4701924bd2d73d7f54)): ?>
<?php $component = $__componentOriginal9ac128a9029c0e4701924bd2d73d7f54; ?>
<?php unset($__componentOriginal9ac128a9029c0e4701924bd2d73d7f54); ?>
<?php endif; ?>
<?php /**PATH E:\ISD\app\resources\views/violations/show.blade.php ENDPATH**/ ?>